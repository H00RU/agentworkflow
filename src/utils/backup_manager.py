"""
Backup Manager: Handle backup operations to Google Drive.
Supports incremental and full backups with compression.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage backups to Google Drive."""

    def __init__(self, drive_path: str, backup_root: str = 'agentworkflow'):
        """
        Initialize backup manager.

        Args:
            drive_path: Path to Google Drive mount point
            backup_root: Root folder in Drive for backups
        """
        self.drive_path = drive_path
        self.backup_root = backup_root
        self.backup_base = os.path.join(drive_path, backup_root)

        # Create backup directory
        os.makedirs(self.backup_base, exist_ok=True)

        logger.info(f"Backup manager initialized: {self.backup_base}")

    def backup_checkpoints(self, source_dir: str, checkpoint_name: str = None) -> bool:
        """
        Backup checkpoint directory to Drive.

        Args:
            source_dir: Source checkpoint directory
            checkpoint_name: Name for backup (timestamp if None)

        Returns:
            True if successful
        """
        if not os.path.exists(source_dir):
            logger.error(f"Source directory not found: {source_dir}")
            return False

        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{timestamp}"

        dest_dir = os.path.join(self.backup_base, 'checkpoints', checkpoint_name)

        try:
            os.makedirs(dest_dir, exist_ok=True)

            # Copy files
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    src = os.path.join(root, file)
                    rel_path = os.path.relpath(src, source_dir)
                    dest = os.path.join(dest_dir, rel_path)

                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(src, dest)

            logger.info(f"Checkpoint backed up to {dest_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup checkpoint: {e}")
            return False

    def backup_results(self, results: Dict[str, Any], result_name: str = None) -> bool:
        """
        Backup training results to Drive.

        Args:
            results: Results dictionary
            result_name: Name for backup (timestamp if None)

        Returns:
            True if successful
        """
        if result_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_name = f"results_{timestamp}.json"

        dest_dir = os.path.join(self.backup_base, 'results')
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, result_name)

        try:
            with open(dest_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results backed up to {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup results: {e}")
            return False

    def backup_logs(self, log_file: str) -> bool:
        """
        Backup training logs to Drive.

        Args:
            log_file: Path to log file

        Returns:
            True if successful
        """
        if not os.path.exists(log_file):
            logger.error(f"Log file not found: {log_file}")
            return False

        dest_dir = os.path.join(self.backup_base, 'logs')
        os.makedirs(dest_dir, exist_ok=True)

        filename = os.path.basename(log_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_name = f"{os.path.splitext(filename)[0]}_{timestamp}.log"
        dest_path = os.path.join(dest_dir, dest_name)

        try:
            shutil.copy2(log_file, dest_path)
            logger.info(f"Logs backed up to {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup logs: {e}")
            return False

    def backup_directory(self, source_dir: str, relative_path: str = None) -> bool:
        """
        Backup entire directory to Drive.

        Args:
            source_dir: Source directory to backup
            relative_path: Relative path in backup (uses basename if None)

        Returns:
            True if successful
        """
        if not os.path.exists(source_dir):
            logger.error(f"Source directory not found: {source_dir}")
            return False

        if relative_path is None:
            relative_path = os.path.basename(source_dir)

        dest_dir = os.path.join(self.backup_base, relative_path)

        try:
            # Remove destination if exists
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)

            # Copy directory
            shutil.copytree(source_dir, dest_dir)

            logger.info(f"Directory backed up to {dest_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup directory: {e}")
            return False

    def get_backup_status(self) -> Dict[str, Any]:
        """
        Get backup status and file count.

        Returns:
            Status information
        """
        status = {
            'backup_root': self.backup_base,
            'exists': os.path.exists(self.backup_base),
            'subdirectories': {},
        }

        if not status['exists']:
            return status

        for item in os.listdir(self.backup_base):
            item_path = os.path.join(self.backup_base, item)
            if os.path.isdir(item_path):
                file_count = sum(
                    1 for root, dirs, files in os.walk(item_path)
                    for _ in files
                )
                status['subdirectories'][item] = {
                    'path': item_path,
                    'file_count': file_count,
                }

        return status

    def cleanup_old_backups(self, max_backups: int = 5, backup_type: str = 'checkpoints') -> int:
        """
        Remove old backups, keeping only the latest N.

        Args:
            max_backups: Maximum number of backups to keep
            backup_type: Type of backup to cleanup ('checkpoints', 'results', 'logs')

        Returns:
            Number of backups removed
        """
        backup_dir = os.path.join(self.backup_base, backup_type)

        if not os.path.exists(backup_dir):
            return 0

        # Get all backup directories/files sorted by modification time
        backups = []
        for item in os.listdir(backup_dir):
            item_path = os.path.join(backup_dir, item)
            mtime = os.path.getmtime(item_path)
            backups.append((mtime, item_path))

        # Sort by modification time (oldest first)
        backups.sort()

        # Remove old backups
        removed = 0
        for _, backup_path in backups[:-max_backups]:
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
                else:
                    os.remove(backup_path)
                removed += 1
                logger.info(f"Removed old backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to remove backup {backup_path}: {e}")

        return removed

    def list_backups(self, backup_type: str = 'checkpoints') -> List[str]:
        """
        List all backups of a specific type.

        Args:
            backup_type: Type of backup ('checkpoints', 'results', 'logs')

        Returns:
            List of backup names
        """
        backup_dir = os.path.join(self.backup_base, backup_type)

        if not os.path.exists(backup_dir):
            return []

        return sorted(os.listdir(backup_dir))

    def __repr__(self) -> str:
        return f"BackupManager(drive_path={self.drive_path})"
