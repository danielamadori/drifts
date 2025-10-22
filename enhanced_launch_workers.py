#!/usr/bin/env python3
"""
Multi-Worker Launcher Script with Pure YAML Configuration

Launch multiple instances of worker scripts using YAML configuration files.
Pure mode: either use your complete config file AS-IS, or use built-in defaults (no merging).

Usage:
    python enhanced_launch_workers.py start                      # Use default config
    python enhanced_launch_workers.py start --config my_config.yaml
    python enhanced_launch_workers.py start --profile production  # Use specific profile
    python enhanced_launch_workers.py stop                       # Stop all workers
    python enhanced_launch_workers.py status                     # Check worker status
    python enhanced_launch_workers.py logs 1                     # View logs for worker 1
"""

import argparse
import subprocess
import os
import sys
import time
import datetime
import json
import signal
import threading
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    'redis': {
        'host': 'localhost',
        'port': 6379
    },
    'workers': {
        'default': {
            'script': 'worker_cache.py',
            'count': 1,
            'args': []
        }
    },
    'logging': {
        'directory': 'logs',
        'cleanup_days': 7
    },
    'profiles': {
        'development': {
            'workers': {
                'cache_workers': {
                    'script': 'worker_cache.py',
                    'count': 2,
                    'args': ['--verbose']
                }
            }
        },
        'production': {
            'workers': {
                'cache_workers': {
                    'script': 'worker_cache.py',
                    'count': 8,
                    'args': []
                },
                'rcheck_workers': {
                    'script': 'worker_rcheck.py',
                    'count': 4,
                    'args': []
                }
            }
        }
    }
}

class WorkerManager:
    def __init__(self, config_path: Optional[str] = None):
        self.workers_dir = Path("workers")
        self.logs_dir = Path("logs")
        self.pids_file = self.workers_dir / "worker_pids.json"
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Update logs directory from config
        self.logs_dir = Path(self.config.get('logging', {}).get('directory', 'logs'))
        
        # Create directories
        self.workers_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Available worker scripts
        self.available_workers = {
            'worker_cache': 'worker_cache.py',
            'worker_rcheck': 'worker_rcheck.py'
        }
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration - either from specified file OR use defaults (no merging)"""
        
        if config_path:
            # User specified a config file - use ONLY that file
            config_file = Path(config_path)
            
            if not config_file.exists():
                raise FileNotFoundError(f"❌ Specified config file not found: {config_file}")
            
            print(f"📄 Loading configuration from {config_file}")
            
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"❌ Invalid YAML syntax in {config_file}: {e}")
            except Exception as e:
                raise ValueError(f"❌ Error reading config file {config_file}: {e}")
            
            # Validate the config has required structure
            self.validate_config(loaded_config, config_file)
            
            print(f"✅ Using configuration from {config_file} (pure mode)")
            return loaded_config
        
        else:
            # No config file specified - check for auto-detected files
            possible_configs = [
                Path('worker_config.yaml'),
                Path('config.yaml'),
                Path('workers.yaml')
            ]
            
            for possible in possible_configs:
                if possible.exists():
                    print(f"📄 Auto-detected configuration file: {possible}")
                    try:
                        with open(possible, 'r') as f:
                            loaded_config = yaml.safe_load(f) or {}
                        
                        self.validate_config(loaded_config, possible)
                        print(f"✅ Using auto-detected configuration from {possible}")
                        return loaded_config
                    
                    except Exception as e:
                        print(f"⚠️  Error loading auto-detected config {possible}: {e}")
                        continue
            
            # No config file found or specified - use built-in defaults
            print("🔄 Using built-in default configuration")
            return DEFAULT_CONFIG.copy()

    def validate_config(self, config: Dict[str, Any], config_file: Path):
        """Validate that config file has proper structure"""
        errors = []
        
        # Check required top-level sections exist if profiles are defined
        if 'profiles' in config:
            for profile_name, profile_config in config['profiles'].items():
                if not isinstance(profile_config, dict):
                    errors.append(f"Profile '{profile_name}' must be a dictionary")
                    continue
                
                # Check if profile has workers section
                if 'workers' in profile_config:
                    workers = profile_config['workers']
                    if not isinstance(workers, dict):
                        errors.append(f"Profile '{profile_name}': workers must be a dictionary")
                        continue
                    
                    # Validate each worker group
                    for worker_name, worker_config in workers.items():
                        if not isinstance(worker_config, dict):
                            errors.append(f"Profile '{profile_name}': worker '{worker_name}' must be a dictionary")
                            continue
                        
                        # Check required worker fields
                        if 'script' in worker_config:
                            script = worker_config['script']
                            if not isinstance(script, str):
                                errors.append(f"Profile '{profile_name}': worker '{worker_name}': script must be a string")
                        
                        if 'count' in worker_config:
                            count = worker_config['count']
                            if not isinstance(count, int) or count < 1:
                                errors.append(f"Profile '{profile_name}': worker '{worker_name}': count must be a positive integer")
                        
                        if 'args' in worker_config:
                            args = worker_config['args']
                            if not isinstance(args, list):
                                errors.append(f"Profile '{profile_name}': worker '{worker_name}': args must be a list")
        
        # Check redis section if present
        if 'redis' in config:
            redis_config = config['redis']
            if not isinstance(redis_config, dict):
                errors.append("'redis' section must be a dictionary")
            else:
                if 'host' in redis_config and not isinstance(redis_config['host'], str):
                    errors.append("redis.host must be a string")
                if 'port' in redis_config and not isinstance(redis_config['port'], int):
                    errors.append("redis.port must be an integer")
        
        # Check logging section if present
        if 'logging' in config:
            logging_config = config['logging']
            if not isinstance(logging_config, dict):
                errors.append("'logging' section must be a dictionary")
            else:
                if 'directory' in logging_config and not isinstance(logging_config['directory'], str):
                    errors.append("logging.directory must be a string")
                if 'cleanup_days' in logging_config and not isinstance(logging_config['cleanup_days'], int):
                    errors.append("logging.cleanup_days must be an integer")
        
        # If there are validation errors, raise them
        if errors:
            error_msg = f"❌ Configuration validation errors in {config_file}:\n"
            for i, error in enumerate(errors, 1):
                error_msg += f"  {i}. {error}\n"
            raise ValueError(error_msg)
    
    def save_default_config(self, output_path: str = "worker_config.yaml"):
        """Save the default configuration to a YAML file"""
        config_file = Path(output_path)
        
        print(f"📝 Saving default configuration to {config_file}")
        with open(config_file, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, indent=2)
        
        print(f"✅ Configuration saved to {config_file}")
        return True
    
    def get_timestamp(self):
        """Get formatted timestamp for log files"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_log_filename(self, worker_id, worker_script, timestamp):
        """Generate log filename"""
        script_name = Path(worker_script).stem
        return self.logs_dir / f"{script_name}_worker_{worker_id}_{timestamp}.log"
    
    def load_pids(self):
        """Load worker PIDs from file"""
        if self.pids_file.exists():
            try:
                with open(self.pids_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_pids(self, pids_data):
        """Save worker PIDs to file"""
        with open(self.pids_file, 'w') as f:
            json.dump(pids_data, f, indent=2)
    
    def is_process_running(self, pid):
        """Check if a process is still running"""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    
    def get_worker_config(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Get worker configuration for a profile or default"""
        if profile:
            profiles = self.config.get('profiles', {})
            if profile not in profiles:
                available_profiles = list(profiles.keys()) if profiles else ['none']
                raise ValueError(f"❌ Profile '{profile}' not found. Available profiles: {available_profiles}")
            
            profile_config = profiles[profile]
            workers = profile_config.get('workers', {})
            
            if not workers:
                raise ValueError(f"❌ Profile '{profile}' has no workers defined")
            
            print(f"🎯 Using profile: {profile}")
            return workers
        else:
            # No profile specified, use default workers
            default_workers = self.config.get('workers', {})
            if not default_workers:
                raise ValueError("❌ No default workers defined and no profile specified")
            
            return default_workers
    
    def start_workers_from_config(self, profile: Optional[str] = None, worker_groups: Optional[List[str]] = None):
        """Start workers based on configuration"""
        worker_configs = self.get_worker_config(profile)
        
        if not worker_configs:
            print("❌ No worker configurations found")
            return False
        
        # Filter worker groups if specified
        if worker_groups:
            filtered_configs = {k: v for k, v in worker_configs.items() if k in worker_groups}
            if not filtered_configs:
                print(f"❌ No matching worker groups found: {worker_groups}")
                print(f"💡 Available groups: {list(worker_configs.keys())}")
                return False
            worker_configs = filtered_configs
        
        print(f"🚀 Starting workers from configuration")
        print(f"📊 Worker groups: {list(worker_configs.keys())}")
        
        timestamp = self.get_timestamp()
        pids_data = self.load_pids()
        
        # Clean up old PIDs
        active_workers = {}
        for worker_id, worker_info in pids_data.items():
            if self.is_process_running(worker_info['pid']):
                active_workers[worker_id] = worker_info
        
        started_workers = []
        
        # Get Redis configuration - profile-specific first, then top-level, then defaults
        redis_config = {'host': 'localhost', 'port': 6379}  # Default fallback
        
        if profile and profile in self.config.get('profiles', {}):
            # Use profile-specific Redis config if available
            profile_config = self.config['profiles'][profile]
            if 'redis' in profile_config:
                redis_config = profile_config['redis']
                print(f"🔗 Using Redis from profile '{profile}': {redis_config['host']}:{redis_config['port']}")
            else:
                # Fall back to top-level Redis config
                redis_config = self.config.get('redis', redis_config)
                print(f"🔗 Using top-level Redis config: {redis_config['host']}:{redis_config['port']}")
        else:
            # No profile or profile doesn't exist, use top-level Redis config
            redis_config = self.config.get('redis', redis_config)
            print(f"🔗 Using top-level Redis config: {redis_config['host']}:{redis_config['port']}")
        
        for group_name, group_config in worker_configs.items():
            script = group_config.get('script', 'worker_cache.py')
            count = group_config.get('count', 1)
            base_args = group_config.get('args', [])
            
            print(f"\n🔧 Starting {count} instances of {script} (group: {group_name})")
            
            # Check if script exists
            if not os.path.exists(script):
                print(f"❌ Worker script '{script}' not found, skipping group {group_name}")
                continue
            
            for i in range(count):
                worker_id = f"{group_name}_{i+1}"
                # Ensure unique worker ID
                counter = 1
                original_worker_id = worker_id
                while worker_id in active_workers:
                    worker_id = f"{original_worker_id}_{counter}"
                    counter += 1
                
                log_file = self.get_log_filename(worker_id, script, timestamp)
                
                print(f"  🔧 Starting {worker_id}...")
                print(f"     📝 Log file: {log_file}")
                
                # Prepare command with Redis configuration and custom args
                cmd = [sys.executable, script]
                cmd.extend(['--redis-host', redis_config.get('host', 'localhost')])
                cmd.extend(['--redis-port', str(redis_config.get('port', 6379))])
                cmd.extend(base_args)
                
                try:
                    # Create log file with header
                    with open(log_file, 'w') as log_f:
                        log_f.write(f"=== Worker {worker_id} Started ===\n")
                        log_f.write(f"Group: {group_name}\n")
                        log_f.write(f"Script: {script}\n")
                        log_f.write(f"Command: {' '.join(cmd)}\n")
                        log_f.write(f"Redis: {redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}\n")
                        log_f.write(f"Started at: {datetime.datetime.now().isoformat()}\n")
                        log_f.write(f"{'='*50}\n\n")
                        log_f.flush()
                    
                    # Start process
                    process = subprocess.Popen(
                        cmd,
                        stdout=open(log_file, 'a'),
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd()
                    )
                    
                    # Store worker info
                    worker_info = {
                        'pid': process.pid,
                        'worker_script': script,
                        'worker_group': group_name,
                        'log_file': str(log_file),
                        'started_at': datetime.datetime.now().isoformat(),
                        'command': cmd,
                        'redis_config': redis_config
                    }
                    
                    active_workers[worker_id] = worker_info
                    started_workers.append(worker_id)
                    
                    print(f"     ✅ Started with PID {process.pid}")
                    time.sleep(0.5)  # Small delay between starts
                    
                except Exception as e:
                    print(f"     ❌ Failed to start {worker_id}: {e}")
                    continue
        
        # Save updated PIDs
        self.save_pids(active_workers)
        
        print(f"\n🎉 Successfully started {len(started_workers)} workers")
        print(f"📁 Logs directory: {self.logs_dir.absolute()}")
        print(f"🔧 Worker PIDs saved to: {self.pids_file}")
        
        # Show quick status
        self.show_status()
        
        return len(started_workers) > 0
    
    def start_workers(self, num_workers, worker_script='worker_cache.py', extra_args=None):
        """Legacy method for backward compatibility"""
        print(f"🚀 Starting {num_workers} instances of {worker_script} (legacy mode)")
        
        if not os.path.exists(worker_script):
            available = ', '.join(self.available_workers.values())
            print(f"❌ Worker script '{worker_script}' not found")
            print(f"💡 Available workers: {available}")
            return False
        
        timestamp = self.get_timestamp()
        pids_data = self.load_pids()
        
        # Clean up old PIDs that are no longer running
        active_workers = {}
        for worker_id, worker_info in pids_data.items():
            if self.is_process_running(worker_info['pid']):
                active_workers[worker_id] = worker_info
        
        started_workers = []
        redis_config = self.config.get('redis', {'host': 'localhost', 'port': 6379})
        print(f"🔗 Using Redis config: {redis_config['host']}:{redis_config['port']}")
        
        for i in range(num_workers):
            worker_id = f"worker_{len(active_workers) + i + 1}"
            log_file = self.get_log_filename(worker_id, worker_script, timestamp)
            
            print(f"  🔧 Starting {worker_id}...")
            print(f"     📝 Log file: {log_file}")
            
            # Prepare command
            cmd = [sys.executable, worker_script]
            cmd.extend(['--redis-host', redis_config.get('host', 'localhost')])
            cmd.extend(['--redis-port', str(redis_config.get('port', 6379))])
            if extra_args:
                cmd.extend(extra_args)
            
            try:
                # Start process with log redirection
                with open(log_file, 'w') as log_f:
                    log_f.write(f"=== Worker {worker_id} Started ===\n")
                    log_f.write(f"Command: {' '.join(cmd)}\n")
                    log_f.write(f"Started at: {datetime.datetime.now().isoformat()}\n")
                    log_f.write(f"{'='*50}\n\n")
                    log_f.flush()
                
                # Start process
                process = subprocess.Popen(
                    cmd,
                    stdout=open(log_file, 'a'),
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd()
                )
                
                # Store worker info
                worker_info = {
                    'pid': process.pid,
                    'worker_script': worker_script,
                    'log_file': str(log_file),
                    'started_at': datetime.datetime.now().isoformat(),
                    'command': cmd,
                    'redis_config': redis_config
                }
                
                active_workers[worker_id] = worker_info
                started_workers.append(worker_id)
                
                print(f"     ✅ Started with PID {process.pid}")
                time.sleep(0.5)  # Small delay between starts
                
            except Exception as e:
                print(f"     ❌ Failed to start {worker_id}: {e}")
                continue
        
        # Save updated PIDs
        self.save_pids(active_workers)
        
        print(f"\n🎉 Successfully started {len(started_workers)} workers")
        return len(started_workers) > 0
    
    def stop_workers(self, worker_ids=None):
        """Stop worker instances"""
        pids_data = self.load_pids()
        
        if not pids_data:
            print("📭 No workers found")
            return True
        
        if worker_ids:
            # Stop specific workers
            workers_to_stop = {wid: pids_data[wid] for wid in worker_ids if wid in pids_data}
        else:
            # Stop all workers
            workers_to_stop = pids_data
        
        if not workers_to_stop:
            print("📭 No matching workers found")
            return True
        
        print(f"🛑 Stopping {len(workers_to_stop)} workers...")
        
        stopped_count = 0
        remaining_workers = {}
        
        for worker_id, worker_info in pids_data.items():
            if worker_id in workers_to_stop:
                pid = worker_info['pid']
                print(f"  🛑 Stopping {worker_id} (PID {pid})...")
                
                try:
                    if self.is_process_running(pid):
                        os.kill(pid, signal.SIGTERM)
                        
                        # Wait for graceful shutdown
                        for _ in range(10):  # Wait up to 10 seconds
                            time.sleep(1)
                            if not self.is_process_running(pid):
                                break
                        
                        # Force kill if still running
                        if self.is_process_running(pid):
                            print(f"     ⚠️  Force killing {worker_id}...")
                            os.kill(pid, signal.SIGKILL)
                        
                        print(f"     ✅ Stopped {worker_id}")
                        stopped_count += 1
                    else:
                        print(f"     ℹ️  {worker_id} was not running")
                        stopped_count += 1
                        
                except Exception as e:
                    print(f"     ❌ Error stopping {worker_id}: {e}")
                    remaining_workers[worker_id] = worker_info
            else:
                # Keep this worker
                remaining_workers[worker_id] = worker_info
        
        # Save remaining workers
        self.save_pids(remaining_workers)
        
        print(f"\n✅ Stopped {stopped_count} workers")
        return True
    
    def show_status(self):
        """Show status of all workers"""
        pids_data = self.load_pids()
        
        if not pids_data:
            print("📭 No workers registered")
            return
        
        print(f"\n📊 Worker Status ({len(pids_data)} workers)")
        print("=" * 100)
        
        active_count = 0
        inactive_count = 0
        groups = {}
        
        for worker_id, worker_info in sorted(pids_data.items()):
            pid = worker_info['pid']
            script = worker_info['worker_script']
            group = worker_info.get('worker_group', 'legacy')
            started_at = worker_info.get('started_at', 'unknown')
            log_file = worker_info.get('log_file', 'unknown')
            redis_config = worker_info.get('redis_config', {'host': 'unknown', 'port': 'unknown'})
            
            if self.is_process_running(pid):
                status = "🟢 RUNNING"
                active_count += 1
                groups[group] = groups.get(group, {'active': 0, 'inactive': 0})
                groups[group]['active'] += 1
            else:
                status = "🔴 STOPPED"
                inactive_count += 1
                groups[group] = groups.get(group, {'active': 0, 'inactive': 0})
                groups[group]['inactive'] += 1
            
            redis_info = f"{redis_config.get('host', 'unknown')}:{redis_config.get('port', 'unknown')}"
            
            print(f"{worker_id:15} | {status} | PID {pid:6} | {script:20} | {group:15} | Redis: {redis_info}")
            print(f"               | Started: {started_at}")
            print(f"               | Log: {Path(log_file).name}")
            print("-" * 100)
        
        print(f"\nSummary: {active_count} active, {inactive_count} inactive")
        
        if groups:
            print(f"\nBy Group:")
            for group, counts in groups.items():
                print(f"  {group}: {counts['active']} active, {counts['inactive']} inactive")
    
    def view_logs(self, worker_id=None, tail=False, follow=False, lines=50):
        """View worker logs"""
        pids_data = self.load_pids()
        
        if worker_id and worker_id not in pids_data:
            print(f"❌ Worker '{worker_id}' not found")
            return False
        
        if worker_id:
            # View specific worker log
            log_file = Path(pids_data[worker_id]['log_file'])
            if not log_file.exists():
                print(f"❌ Log file not found: {log_file}")
                return False
            
            print(f"📋 Viewing log for {worker_id}: {log_file.name}")
            
            if follow:
                # Follow log in real-time
                self._tail_follow(log_file)
            elif tail:
                # Show last N lines
                self._show_tail(log_file, lines)
            else:
                # Show entire log
                with open(log_file, 'r') as f:
                    print(f.read())
        
        else:
            # View all logs
            if not pids_data:
                print("📭 No workers found")
                return False
            
            for wid, worker_info in sorted(pids_data.items()):
                log_file = Path(worker_info['log_file'])
                if log_file.exists():
                    print(f"\n{'='*20} {wid} {'='*20}")
                    if tail:
                        self._show_tail(log_file, lines)
                    else:
                        with open(log_file, 'r') as f:
                            content = f.read()
                            if len(content) > 2000:  # Truncate very long logs
                                print(content[:1000])
                                print(f"\n... [truncated {len(content)-2000} characters] ...\n")
                                print(content[-1000:])
                            else:
                                print(content)
        
        return True
    
    def _show_tail(self, log_file, lines=50):
        """Show last N lines of a file"""
        try:
            with open(log_file, 'r') as f:
                content_lines = f.readlines()
                if len(content_lines) > lines:
                    print(f"... [showing last {lines} lines] ...")
                for line in content_lines[-lines:]:
                    print(line, end='')
        except Exception as e:
            print(f"❌ Error reading log: {e}")
    
    def _tail_follow(self, log_file):
        """Follow log file in real-time (like tail -f)"""
        print(f"📡 Following {log_file.name} (Press Ctrl+C to stop)")
        
        try:
            with open(log_file, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        print(line, end='')
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n📴 Stopped following log")
    
    def cleanup_old_logs(self, days=None):
        """Clean up old log files"""
        if days is None:
            days = self.config.get('logging', {}).get('cleanup_days', 7)
        
        cutoff_time = time.time() - (days * 24 * 3600)
        cleaned_count = 0
        
        print(f"🧹 Cleaning up logs older than {days} days...")
        
        for log_file in self.logs_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                cleaned_count += 1
                print(f"  🗑️  Deleted {log_file.name}")
        
        print(f"✅ Cleaned up {cleaned_count} old log files")

    def clean_directories(self):
        """Clean and recreate logs and workers directories"""
        import shutil
        
        print("🧹 Cleaning directories...")
        
        # Clean logs directory
        if self.logs_dir.exists():
            print(f"  🗑️  Removing logs directory: {self.logs_dir}")
            shutil.rmtree(self.logs_dir)
        
        self.logs_dir.mkdir(exist_ok=True)
        print(f"  ✅ Recreated logs directory: {self.logs_dir}")
        
        # Clean workers directory (including PID file)
        if self.workers_dir.exists():
            print(f"  🗑️  Removing workers directory: {self.workers_dir}")
            shutil.rmtree(self.workers_dir)
        
        self.workers_dir.mkdir(exist_ok=True)
        print(f"  ✅ Recreated workers directory: {self.workers_dir}")
        
        print("✅ Directories cleaned and recreated")
    
    def clean_restart(self, profile: Optional[str] = None, worker_groups: Optional[List[str]] = None):
        """Stop all workers, clean directories, and start fresh"""
        print("🔄 Starting clean restart sequence...")
        print("=" * 60)
        
        # Step 1: Stop all workers
        print("\n📍 Step 1/3: Stopping all workers")
        self.stop_workers()
        
        # Step 2: Clean directories
        print("\n📍 Step 2/3: Cleaning directories")
        self.clean_directories()
        
        # Step 3: Start workers
        print("\n📍 Step 3/3: Starting workers")
        success = self.start_workers_from_config(profile, worker_groups)
        
        if success:
            print("\n🎉 Clean restart completed successfully!")
        else:
            print("\n⚠️  Clean restart completed with some failures")
        
        return success
    
    def show_config(self):
        """Show current configuration"""
        print("📄 Current Configuration:")
        print("=" * 50)
        print(yaml.dump(self.config, default_flow_style=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Launch and manage multiple worker instances with pure YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start workers using default config
  python launch_workers.py start
  
  # Start workers using specific config file (pure mode)
  python launch_workers.py --config myconfig.yaml start
  
  # Start workers using a specific profile
  python launch_workers.py start --profile production
  python launch_workers.py --config myconfig.yaml start --profile production
  
  # Start specific worker groups
  python launch_workers.py start --groups cache_workers rcheck_workers
  
  # Clean restart: stop all workers, clean directories, and start fresh
  python launch_workers.py clean-restart
  python launch_workers.py clean-restart --profile production
  python launch_workers.py --config myconfig.yaml clean-restart --profile production
  
  # Legacy mode: start 4 instances manually
  python launch_workers.py start-legacy 4 --worker worker_cache.py
  
  # Check status
  python launch_workers.py status
  
  # View logs
  python launch_workers.py logs worker_1
  python launch_workers.py logs --tail
  
  # Stop workers
  python launch_workers.py stop
  
  # Generate default config file
  python launch_workers.py init-config
        """
    )
    
    # Global options
    parser.add_argument('--config', help='YAML configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command (new YAML-based)
    start_parser = subparsers.add_parser('start', help='Start workers from configuration')
    start_parser.add_argument('--profile', help='Configuration profile to use')
    start_parser.add_argument('--groups', nargs='+', help='Specific worker groups to start')
    
    # Start legacy command
    start_legacy_parser = subparsers.add_parser('start-legacy', help='Start workers (legacy mode)')
    start_legacy_parser.add_argument('num_workers', type=int, help='Number of workers to start')
    start_legacy_parser.add_argument('--worker', default='worker_cache.py', 
                                   help='Worker script to run (default: worker_cache.py)')
    start_legacy_parser.add_argument('--args', help='Additional arguments to pass to worker script')
    
    # Clean restart command - NEW!
    restart_parser = subparsers.add_parser('clean-restart', help='Stop workers, clean directories, and start fresh')
    restart_parser.add_argument('--profile', help='Configuration profile to use')
    restart_parser.add_argument('--groups', nargs='+', help='Specific worker groups to start')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop worker instances')
    stop_parser.add_argument('worker_ids', nargs='*', help='Specific worker IDs to stop (default: all)')
    
    # Status command
    subparsers.add_parser('status', help='Show worker status')
    
    # Config command
    subparsers.add_parser('show-config', help='Show current configuration')
    
    # Init config command
    init_parser = subparsers.add_parser('init-config', help='Generate default configuration file')
    init_parser.add_argument('--output', default='worker_config.yaml', 
                           help='Output file path (default: worker_config.yaml)')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View worker logs')
    logs_parser.add_argument('worker_id', nargs='?', help='Specific worker ID (default: all)')
    logs_parser.add_argument('--tail', action='store_true', help='Show only last lines')
    logs_parser.add_argument('--follow', action='store_true', help='Follow log in real-time')
    logs_parser.add_argument('--lines', type=int, default=50, help='Number of lines to show (with --tail)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old log files')
    cleanup_parser.add_argument('--days', type=int, help='Delete logs older than N days (default: from config)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'init-config':
            manager = WorkerManager()
            manager.save_default_config(args.output)
            return 0
        
        manager = WorkerManager(args.config)
        
        if args.command == 'start':
            success = manager.start_workers_from_config(args.profile, args.groups)
            return 0 if success else 1
        
        elif args.command == 'clean-restart':
            success = manager.clean_restart(args.profile, args.groups)
            return 0 if success else 1
            
        elif args.command == 'start-legacy':
            extra_args = args.args.split() if args.args else None
            success = manager.start_workers(args.num_workers, args.worker, extra_args)
            return 0 if success else 1
            
        elif args.command == 'stop':
            success = manager.stop_workers(args.worker_ids if args.worker_ids else None)
            return 0 if success else 1
            
        elif args.command == 'status':
            manager.show_status()
            return 0
            
        elif args.command == 'show-config':
            manager.show_config()
            return 0
            
        elif args.command == 'logs':
            success = manager.view_logs(args.worker_id, args.tail, args.follow, args.lines)
            return 0 if success else 1
            
        elif args.command == 'cleanup':
            manager.cleanup_old_logs(args.days)
            return 0
            
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
