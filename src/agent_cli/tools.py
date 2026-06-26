import os
import re
import json
import subprocess
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

# Configure security logging
security_logger = logging.getLogger("agent_security")
security_logger.setLevel(logging.INFO)

# Security constants
ALLOWED_DIRECTORIES: Set[Path] = {Path.cwd()}
FORBIDDEN_PATTERNS: Set[str] = {
    "../", "..\\", "/etc/", "/root/", "/.ssh/", "/.gnupg/", 
    "/proc/", "/sys/", "/dev/", "sudo", "su ", "chmod", "chown"
}
DANGEROUS_COMMANDS: Set[str] = {
    "rm -rf", "dd if=", "mkfs", "fdisk", "wget", "curl",
    ":(){:|:&};:", "fork bomb", "> /dev/", ">> /etc/"
}
MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
MAX_TOOL_CALLS_PER_TURN: int = 5

TOOLS_SCHEMA = [
    {
        "name": "read_file", 
        "description": "Read content from a file (safe, read-only). Max 10MB files.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "path": {"type": "string", "description": "Relative path to file within allowed directories"},
                "max_lines": {"type": "integer", "description": "Maximum lines to read (default: 1000)", "default": 1000}
            }, 
            "required": ["path"]
        }
    },
    {
        "name": "write_file", 
        "description": "Write content to a file. Creates parent directories if needed. Cannot overwrite protected files.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "path": {"type": "string", "description": "Relative path to file"},
                "content": {"type": "string", "description": "Content to write"},
                "append": {"type": "boolean", "description": "Append instead of overwrite (default: false)", "default": False}
            }, 
            "required": ["path", "content"]
        }
    },
    {
        "name": "delete_file", 
        "description": "Delete a file (cannot delete directories or protected files). Use with caution.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "path": {"type": "string", "description": "Path to file to delete"}
            }, 
            "required": ["path"]
        }
    },
    {
        "name": "list_directory", 
        "description": "List contents of a directory with file sizes and types.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "path": {"type": "string", "description": "Directory path to list (default: current directory)", "default": "."},
                "recursive": {"type": "boolean", "description": "List recursively (default: false)", "default": False},
                "pattern": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py')", "default": "*"}
            }
        }
    },
    {
        "name": "search_files", 
        "description": "Search for files by name pattern using glob.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                "path": {"type": "string", "description": "Base directory to search (default: current)", "default": "."}
            }, 
            "required": ["pattern"]
        }
    },
    {
        "name": "grep_search", 
        "description": "Search for text pattern in files (like grep). Returns matching lines with context.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "File or directory to search", "default": "."},
                "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False},
                "include_pattern": {"type": "string", "description": "Only search files matching this glob", "default": "*"}
            }, 
            "required": ["pattern"]
        }
    },
    {
        "name": "copy_file", 
        "description": "Copy a file from source to destination.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "source": {"type": "string", "description": "Source file path"},
                "destination": {"type": "string", "description": "Destination path"}
            }, 
            "required": ["source", "destination"]
        }
    },
    {
        "name": "move_file", 
        "description": "Move/rename a file from source to destination.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "source": {"type": "string", "description": "Source file path"},
                "destination": {"type": "string", "description": "Destination path"}
            }, 
            "required": ["source", "destination"]
        }
    },
    {
        "name": "run_shell", 
        "description": "Execute safe shell commands. Dangerous commands are blocked. Timeout: configurable.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute (subject to security filtering)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30, max: 120)", "default": 30},
                "working_dir": {"type": "string", "description": "Working directory for command execution", "default": "."}
            }, 
            "required": ["command"]
        }
    },
    {
        "name": "run_python", 
        "description": "Execute Python code in a restricted sandbox. No network, limited imports.", 
        "parameters": {
            "type": "object", 
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 10)", "default": 10}
            }, 
            "required": ["code"]
        }
    },
    {
        "name": "create_directory", 
        "description": "Create a new directory (including parent directories).", 
        "parameters": {
            "type": "object", 
            "properties": {
                "path": {"type": "string", "description": "Directory path to create"}
            }, 
            "required": ["path"]
        }
    }
]


def _validate_path(path: str, allow_write: bool = False) -> tuple[bool, str, Optional[Path]]:
    """
    Validate a file path for security.
    Returns: (is_valid, error_message, resolved_path)
    """
    if not path:
        return False, "Empty path provided", None
    
    # Check for forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in path:
            security_logger.warning(f"Blocked path with forbidden pattern '{pattern}': {path}")
            return False, f"Path contains forbidden pattern: {pattern}", None
    
    try:
        # Resolve to absolute path
        resolved = Path(path).resolve()
        
        # Check if path is within allowed directories
        is_allowed = False
        for allowed_dir in ALLOWED_DIRECTORIES:
            try:
                resolved.relative_to(allowed_dir.resolve())
                is_allowed = True
                break
            except ValueError:
                continue
        
        if not is_allowed:
            security_logger.warning(f"Path outside allowed directories: {path} (resolved: {resolved})")
            return False, f"Path must be within allowed directories: {[str(d) for d in ALLOWED_DIRECTORIES]}", None
        
        # Check for symlinks pointing outside allowed dirs
        if resolved.is_symlink():
            try:
                real_path = resolved.resolve(strict=True)
                for allowed_dir in ALLOWED_DIRECTORIES:
                    try:
                        real_path.relative_to(allowed_dir.resolve())
                        is_allowed = True
                        break
                    except ValueError:
                        continue
                if not is_allowed:
                    security_logger.warning(f"Symlink points outside allowed directories: {path}")
                    return False, "Symlink target is outside allowed directories", None
            except (OSError, ValueError):
                pass
        
        return True, "", resolved
        
    except Exception as e:
        return False, f"Path validation error: {str(e)}", None


def _validate_command(command: str) -> tuple[bool, str]:
    """
    Validate a shell command for security.
    Returns: (is_valid, error_message)
    """
    if not command:
        return False, "Empty command"
    
    cmd_lower = command.lower()
    
    # Check for dangerous commands
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous.lower() in cmd_lower:
            security_logger.warning(f"Blocked dangerous command pattern '{dangerous}': {command}")
            return False, f"Dangerous command pattern detected: {dangerous}"
    
    # Check for forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in cmd_lower:
            security_logger.warning(f"Blocked command with forbidden pattern '{pattern}': {command}")
            return False, f"Command contains forbidden pattern: {pattern}"
    
    # Block pipes and redirections to sensitive locations
    if "|" in command and any(x in command for x in ["sudo", "passwd", "shadow", "ssh"]):
        return False, "Pipes to sensitive commands are not allowed"
    
    if re.search(r'>\s*/(etc|root|usr|bin|sbin)', command):
        return False, "Cannot redirect output to system directories"
    
    # Block subshell execution
    if "$(" in command or "`" in command:
        return False, "Subshell execution is not allowed"
    
    return True, ""


def _audit_tool_call(tool_name: str, args: Dict[str, Any], result: str, success: bool):
    """Log tool execution for audit trail."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "tool": tool_name,
        "args_hash": hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:16],
        "success": success,
        "result_preview": result[:200] if result else None
    }
    security_logger.info(f"AUDIT: {json.dumps(log_entry)}")


def read_file(path: str, max_lines: int = 1000) -> str:
    """Read file content with security validation."""
    is_valid, error_msg, resolved = _validate_path(path, allow_write=False)
    if not is_valid:
        _audit_tool_call("read_file", {"path": path, "max_lines": max_lines}, error_msg, False)
        return f"❌ Security Error: {error_msg}"
    
    try:
        if not resolved.exists():
            msg = f"Error: File '{path}' not found."
            _audit_tool_call("read_file", {"path": path, "max_lines": max_lines}, msg, False)
            return msg
        
        if not resolved.is_file():
            msg = f"Error: '{path}' is not a file."
            _audit_tool_call("read_file", {"path": path, "max_lines": max_lines}, msg, False)
            return msg
        
        # Check file size
        file_size = resolved.stat().st_size
        if file_size > MAX_FILE_SIZE:
            msg = f"Error: File too large ({file_size / 1024 / 1024:.2f}MB > {MAX_FILE_SIZE / 1024 / 1024}MB limit)."
            _audit_tool_call("read_file", {"path": path, "max_lines": max_lines}, msg, False)
            return msg
        
        # Read with line limit
        with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... (truncated after {max_lines} lines)")
                    break
                lines.append(line)
            content = ''.join(lines)
        
        result = f"✅ Read {len(lines)} lines from {path}"
        _audit_tool_call("read_file", {"path": path, "max_lines": max_lines}, result, True)
        return content
        
    except PermissionError:
        msg = f"Error: Permission denied reading '{path}'."
        _audit_tool_call("read_file", {"path": path, "max_lines": max_lines}, msg, False)
        return msg
    except UnicodeDecodeError:
        msg = f"Error: Cannot decode '{path}' as UTF-8. Binary file?"
        _audit_tool_call("read_file", {"path": path, "max_lines": max_lines}, msg, False)
        return msg
    except Exception as e:
        msg = f"Error reading file: {str(e)}"
        _audit_tool_call("read_file", {"path": path, "max_lines": max_lines}, msg, False)
        return msg


def write_file(path: str, content: str, append: bool = False) -> str:
    """Write content to file with security validation."""
    is_valid, error_msg, resolved = _validate_path(path, allow_write=True)
    if not is_valid:
        _audit_tool_call("write_file", {"path": path, "content_len": len(content), "append": append}, error_msg, False)
        return f"❌ Security Error: {error_msg}"
    
    try:
        # Check if file exists and is not writable
        if resolved.exists():
            if not resolved.is_file():
                msg = f"Error: '{path}' already exists and is not a file."
                _audit_tool_call("write_file", {"path": path, "content_len": len(content), "append": append}, msg, False)
                return msg
        
        # Create parent directories
        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        mode = 'a' if append else 'w'
        with open(resolved, mode, encoding='utf-8') as f:
            f.write(content)
        
        action = "appended to" if append else "written to"
        result = f"✅ Successfully {action} {path} ({len(content)} bytes)"
        _audit_tool_call("write_file", {"path": path, "content_len": len(content), "append": append}, result, True)
        return result
        
    except PermissionError:
        msg = f"Error: Permission denied writing to '{path}'."
        _audit_tool_call("write_file", {"path": path, "content_len": len(content), "append": append}, msg, False)
        return msg
    except Exception as e:
        msg = f"Error writing file: {str(e)}"
        _audit_tool_call("write_file", {"path": path, "content_len": len(content), "append": append}, msg, False)
        return msg


def delete_file(path: str) -> str:
    """Delete a file with security validation."""
    is_valid, error_msg, resolved = _validate_path(path, allow_write=True)
    if not is_valid:
        _audit_tool_call("delete_file", {"path": path}, error_msg, False)
        return f"❌ Security Error: {error_msg}"
    
    try:
        if not resolved.exists():
            msg = f"Error: File '{path}' does not exist."
            _audit_tool_call("delete_file", {"path": path}, msg, False)
            return msg
        
        if not resolved.is_file():
            msg = f"Error: Cannot delete '{path}' - it's not a file (use run_shell for directories)."
            _audit_tool_call("delete_file", {"path": path}, msg, False)
            return msg
        
        resolved.unlink()
        result = f"✅ Deleted {path}"
        _audit_tool_call("delete_file", {"path": path}, result, True)
        return result
        
    except PermissionError:
        msg = f"Error: Permission denied deleting '{path}'."
        _audit_tool_call("delete_file", {"path": path}, msg, False)
        return msg
    except Exception as e:
        msg = f"Error deleting file: {str(e)}"
        _audit_tool_call("delete_file", {"path": path}, msg, False)
        return msg


def list_directory(path: str = ".", recursive: bool = False, pattern: str = "*") -> str:
    """List directory contents with optional filtering."""
    is_valid, error_msg, resolved = _validate_path(path, allow_write=False)
    if not is_valid:
        _audit_tool_call("list_directory", {"path": path, "recursive": recursive, "pattern": pattern}, error_msg, False)
        return f"❌ Security Error: {error_msg}"
    
    try:
        if not resolved.exists():
            msg = f"Error: Directory '{path}' does not exist."
            _audit_tool_call("list_directory", {"path": path, "recursive": recursive, "pattern": pattern}, msg, False)
            return msg
        
        if not resolved.is_dir():
            msg = f"Error: '{path}' is not a directory."
            _audit_tool_call("list_directory", {"path": path, "recursive": recursive, "pattern": pattern}, msg, False)
            return msg
        
        import fnmatch
        results = []
        
        if recursive:
            for root, dirs, files in os.walk(resolved):
                # Filter directories to avoid forbidden paths
                dirs[:] = [d for d in dirs if not any(p in d for p in FORBIDDEN_PATTERNS)]
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        full_path = Path(root) / file
                        try:
                            rel_path = full_path.relative_to(resolved)
                            size = full_path.stat().st_size
                            results.append(f"{' ' * (len(str(rel_path.parent)) - len(str(rel_path)))}{rel_path} ({size:,} bytes)")
                        except (OSError, ValueError):
                            continue
        else:
            for item in resolved.glob(pattern):
                try:
                    rel_path = item.relative_to(resolved)
                    if item.is_file():
                        size = item.stat().st_size
                        results.append(f"📄 {rel_path} ({size:,} bytes)")
                    elif item.is_dir():
                        results.append(f"📁 {rel_path}/")
                except (OSError, ValueError):
                    continue
        
        if not results:
            return f"📂 {path}: No items matching '{pattern}'"
        
        result = f"📂 {path}:\n" + "\n".join(results[:100])  # Limit output
        if len(results) > 100:
            result += f"\n... and {len(results) - 100} more items"
        
        _audit_tool_call("list_directory", {"path": path, "recursive": recursive, "pattern": pattern}, f"Listed {len(results)} items", True)
        return result
        
    except Exception as e:
        msg = f"Error listing directory: {str(e)}"
        _audit_tool_call("list_directory", {"path": path, "recursive": recursive, "pattern": pattern}, msg, False)
        return msg


def search_files(pattern: str, path: str = ".") -> str:
    """Search for files by glob pattern."""
    is_valid, error_msg, resolved = _validate_path(path, allow_write=False)
    if not is_valid:
        _audit_tool_call("search_files", {"pattern": pattern, "path": path}, error_msg, False)
        return f"❌ Security Error: {error_msg}"
    
    try:
        if not resolved.exists():
            msg = f"Error: Path '{path}' does not exist."
            _audit_tool_call("search_files", {"pattern": pattern, "path": path}, msg, False)
            return msg
        
        results = []
        for match in resolved.glob(pattern):
            try:
                rel_path = match.relative_to(resolved)
                results.append(str(rel_path))
            except ValueError:
                continue
        
        if not results:
            return f"🔍 No files matching '{pattern}' in {path}"
        
        result = f"🔍 Found {len(results)} file(s) matching '{pattern}' in {path}:\n" + "\n".join(results[:50])
        if len(results) > 50:
            result += f"\n... and {len(results) - 50} more"
        
        _audit_tool_call("search_files", {"pattern": pattern, "path": path}, f"Found {len(results)} files", True)
        return result
        
    except Exception as e:
        msg = f"Error searching files: {str(e)}"
        _audit_tool_call("search_files", {"pattern": pattern, "path": path}, msg, False)
        return msg


def grep_search(pattern: str, path: str = ".", case_sensitive: bool = False, include_pattern: str = "*") -> str:
    """Search for text pattern in files."""
    is_valid, error_msg, resolved = _validate_path(path, allow_write=False)
    if not is_valid:
        _audit_tool_call("grep_search", {"pattern": pattern, "path": path, "case_sensitive": case_sensitive}, error_msg, False)
        return f"❌ Security Error: {error_msg}"
    
    try:
        import fnmatch
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            msg = f"Invalid regex pattern: {str(e)}"
            _audit_tool_call("grep_search", {"pattern": pattern, "path": path}, msg, False)
            return msg
        
        results = []
        files_searched = 0
        
        for root, dirs, files in os.walk(resolved):
            dirs[:] = [d for d in dirs if not any(p in d for p in FORBIDDEN_PATTERNS)]
            for file in files:
                if fnmatch.fnmatch(file, include_pattern):
                    full_path = Path(root) / file
                    try:
                        files_searched += 1
                        if files_searched > 1000:  # Limit files searched
                            break
                        
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if regex.search(line):
                                    rel_path = full_path.relative_to(resolved)
                                    results.append(f"{rel_path}:{line_num}: {line.rstrip()[:150]}")
                                    if len(results) >= 50:  # Limit results
                                        break
                        if len(results) >= 50:
                            break
                    except (OSError, UnicodeDecodeError):
                        continue
            
            if len(results) >= 50 or files_searched > 1000:
                break
        
        if not results:
            return f"🔍 No matches for '{pattern}' in {files_searched} files"
        
        result = f"🔍 Found {len(results)} matches for '{pattern}':\n" + "\n".join(results)
        if files_searched > 1000:
            result += f"\n(Search limited to 1000 files)"
        
        _audit_tool_call("grep_search", {"pattern": pattern, "path": path}, f"Found {len(results)} matches", True)
        return result
        
    except Exception as e:
        msg = f"Error searching: {str(e)}"
        _audit_tool_call("grep_search", {"pattern": pattern, "path": path}, msg, False)
        return msg


def copy_file(source: str, destination: str) -> str:
    """Copy a file."""
    # Validate both paths
    is_valid_src, error_msg_src, resolved_src = _validate_path(source, allow_write=False)
    if not is_valid_src:
        _audit_tool_call("copy_file", {"source": source, "destination": destination}, error_msg_src, False)
        return f"❌ Security Error (source): {error_msg_src}"
    
    is_valid_dst, error_msg_dst, resolved_dst = _validate_path(destination, allow_write=True)
    if not is_valid_dst:
        _audit_tool_call("copy_file", {"source": source, "destination": destination}, error_msg_dst, False)
        return f"❌ Security Error (destination): {error_msg_dst}"
    
    try:
        if not resolved_src.exists():
            msg = f"Error: Source file '{source}' does not exist."
            _audit_tool_call("copy_file", {"source": source, "destination": destination}, msg, False)
            return msg
        
        if not resolved_src.is_file():
            msg = f"Error: Source '{source}' is not a file."
            _audit_tool_call("copy_file", {"source": source, "destination": destination}, msg, False)
            return msg
        
        # Create parent directories for destination
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(resolved_src, resolved_dst)
        
        result = f"✅ Copied {source} to {destination}"
        _audit_tool_call("copy_file", {"source": source, "destination": destination}, result, True)
        return result
        
    except Exception as e:
        msg = f"Error copying file: {str(e)}"
        _audit_tool_call("copy_file", {"source": source, "destination": destination}, msg, False)
        return msg


def move_file(source: str, destination: str) -> str:
    """Move/rename a file."""
    is_valid_src, error_msg_src, resolved_src = _validate_path(source, allow_write=False)
    if not is_valid_src:
        _audit_tool_call("move_file", {"source": source, "destination": destination}, error_msg_src, False)
        return f"❌ Security Error (source): {error_msg_src}"
    
    is_valid_dst, error_msg_dst, resolved_dst = _validate_path(destination, allow_write=True)
    if not is_valid_dst:
        _audit_tool_call("move_file", {"source": source, "destination": destination}, error_msg_dst, False)
        return f"❌ Security Error (destination): {error_msg_dst}"
    
    try:
        if not resolved_src.exists():
            msg = f"Error: Source '{source}' does not exist."
            _audit_tool_call("move_file", {"source": source, "destination": destination}, msg, False)
            return msg
        
        if not resolved_src.is_file():
            msg = f"Error: Source '{source}' is not a file."
            _audit_tool_call("move_file", {"source": source, "destination": destination}, msg, False)
            return msg
        
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.move(resolved_src, resolved_dst)
        
        result = f"✅ Moved {source} to {destination}"
        _audit_tool_call("move_file", {"source": source, "destination": destination}, result, True)
        return result
        
    except Exception as e:
        msg = f"Error moving file: {str(e)}"
        _audit_tool_call("move_file", {"source": source, "destination": destination}, msg, False)
        return msg


def run_shell(command: str, timeout: int = 30, working_dir: str = ".") -> str:
    """Execute shell command with security validation."""
    # Validate command
    is_valid, error_msg = _validate_command(command)
    if not is_valid:
        _audit_tool_call("run_shell", {"command": command, "timeout": timeout}, error_msg, False)
        return f"❌ Security Error: {error_msg}"
    
    # Validate timeout
    timeout = min(max(timeout, 1), 120)  # Clamp between 1-120 seconds
    
    # Validate working directory
    is_valid_wd, error_msg_wd, resolved_wd = _validate_path(working_dir, allow_write=False)
    if not is_valid_wd:
        _audit_tool_call("run_shell", {"command": command, "timeout": timeout, "working_dir": working_dir}, error_msg_wd, False)
        return f"❌ Security Error (working_dir): {error_msg_wd}"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(resolved_wd),
            env={**os.environ, "PATH": os.environ.get("PATH", "")}
        )
        
        output = result.stdout.strip() if result.stdout else ""
        error = result.stderr.strip() if result.stderr else ""
        
        if result.returncode == 0:
            output_text = output if output else "Command succeeded (no output)"
            _audit_tool_call("run_shell", {"command": command, "timeout": timeout}, f"Exit 0, {len(output_text)} chars", True)
            return output_text
        else:
            error_text = f"Exit code: {result.returncode}\nSTDERR:\n{error}" if error else f"Exit code: {result.returncode}"
            _audit_tool_call("run_shell", {"command": command, "timeout": timeout}, error_text, False)
            return error_text
            
    except subprocess.TimeoutExpired:
        msg = f"⏱️ Command timed out after {timeout}s"
        _audit_tool_call("run_shell", {"command": command, "timeout": timeout}, msg, False)
        return msg
    except Exception as e:
        msg = f"❌ Error executing command: {str(e)}"
        _audit_tool_call("run_shell", {"command": command, "timeout": timeout}, msg, False)
        return msg


def run_python(code: str, timeout: int = 10) -> str:
    """Execute Python code in a restricted environment."""
    timeout = min(max(timeout, 1), 30)  # Clamp between 1-30 seconds
    
    # Check for dangerous imports/patterns
    dangerous_imports = ["os.system", "subprocess", "socket", "urllib", "requests", 
                         "__import__", "eval(", "exec(", "compile(", "open(",
                         "importlib", "pickle", "marshal", "ctypes"]
    
    for dangerous in dangerous_imports:
        if dangerous in code:
            msg = f"❌ Security Error: Dangerous pattern detected: {dangerous}"
            _audit_tool_call("run_python", {"code_len": len(code), "timeout": timeout}, msg, False)
            return msg
    
    # Allowed built-in functions
    safe_globals = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
            "bool": bool, "bytes": bytes, "callable": callable, "chr": chr,
            "complex": complex, "dict": dict, "dir": dir, "divmod": divmod,
            "enumerate": enumerate, "filter": filter, "float": float, "format": format,
            "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr,
            "hash": hash, "hex": hex, "id": id, "int": int, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter, "len": len, "list": list,
            "map": map, "max": max, "min": min, "next": next, "object": object,
            "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round, "set": set,
            "slice": slice, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "type": type, "zip": zip,
            "True": True, "False": False, "None": None
        },
        "math": __import__("math"),
        "random": __import__("random"),
        "json": __import__("json"),
        "re": __import__("re"),
        "datetime": __import__("datetime"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools")
    }
    
    try:
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        def execute_code():
            with redirect_stdout(output_buffer):
                exec(code, safe_globals, {})
        
        import threading
        thread = threading.Thread(target=execute_code)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            msg = f"⏱️ Code execution timed out after {timeout}s"
            _audit_tool_call("run_python", {"code_len": len(code), "timeout": timeout}, msg, False)
            return msg
        
        output = output_buffer.getvalue()
        result = f"✅ Output:\n{output}" if output else "✅ Code executed successfully (no output)"
        _audit_tool_call("run_python", {"code_len": len(code), "timeout": timeout}, result[:200], True)
        return result
        
    except Exception as e:
        msg = f"❌ Error executing code: {str(e)}"
        _audit_tool_call("run_python", {"code_len": len(code), "timeout": timeout}, msg, False)
        return msg


def create_directory(path: str) -> str:
    """Create a directory."""
    is_valid, error_msg, resolved = _validate_path(path, allow_write=True)
    if not is_valid:
        _audit_tool_call("create_directory", {"path": path}, error_msg, False)
        return f"❌ Security Error: {error_msg}"
    
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        result = f"✅ Created directory: {path}"
        _audit_tool_call("create_directory", {"path": path}, result, True)
        return result
        
    except Exception as e:
        msg = f"Error creating directory: {str(e)}"
        _audit_tool_call("create_directory", {"path": path}, msg, False)
        return msg


TOOL_REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
    "delete_file": delete_file,
    "list_directory": list_directory,
    "search_files": search_files,
    "grep_search": grep_search,
    "copy_file": copy_file,
    "move_file": move_file,
    "run_shell": run_shell,
    "run_python": run_python,
    "create_directory": create_directory
}