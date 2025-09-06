#!/usr/bin/env python3
"""
Migration Script: Legacy to Clean Architecture
Automates the migration from main.py to main_new.py
"""
import os
import shutil
import sys
from datetime import datetime
import subprocess


def backup_current_main():
    """Create backup of current main.py"""
    if os.path.exists('main.py'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'main_legacy_backup_{timestamp}.py'
        shutil.copy2('main.py', backup_name)
        print(f"✅ Backed up main.py to {backup_name}")
        return backup_name
    else:
        print("⚠️  No main.py found to backup")
        return None


def check_dependencies():
    """Check if all required files exist"""
    required_files = [
        'main_new.py',
        'dependency_container.py', 
        'service_providers.py',
        'domain_services.py',
        'api_controllers.py',
        'config.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True


def test_syntax():
    """Test syntax of new architecture files"""
    test_files = [
        'dependency_container.py',
        'service_providers.py', 
        'domain_services.py',
        'api_controllers.py',
        'main_new.py'
    ]
    
    for file in test_files:
        try:
            result = subprocess.run([sys.executable, '-m', 'py_compile', file], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Syntax error in {file}: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error checking {file}: {e}")
            return False
    
    print("✅ All files compile successfully")
    return True


def migrate_main():
    """Replace main.py with main_new.py"""
    try:
        shutil.copy2('main_new.py', 'main.py')
        print("✅ Successfully replaced main.py with clean architecture")
        return True
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False


def test_startup():
    """Test that the new application starts correctly"""
    print("🧪 Testing application startup...")
    try:
        # Import the new main to test initialization
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        print("✅ Application imports successfully")
        return True
    except Exception as e:
        print(f"❌ Application startup test failed: {e}")
        return False


def rollback(backup_file):
    """Rollback to backup if provided"""
    if backup_file and os.path.exists(backup_file):
        shutil.copy2(backup_file, 'main.py')
        print(f"🔄 Rolled back to {backup_file}")
    else:
        print("❌ No backup available for rollback")


def main():
    """Main migration function"""
    print("🚀 Starting migration to clean architecture...")
    print("=" * 50)
    
    # Step 1: Backup current main.py
    backup_file = backup_current_main()
    
    # Step 2: Check dependencies
    if not check_dependencies():
        print("❌ Migration aborted - missing dependencies")
        return False
    
    # Step 3: Test syntax
    if not test_syntax():
        print("❌ Migration aborted - syntax errors found")
        return False
    
    # Step 4: Perform migration
    if not migrate_main():
        print("❌ Migration failed")
        return False
    
    # Step 5: Test startup
    if not test_startup():
        print("⚠️  Migration completed but startup test failed")
        response = input("Do you want to rollback? (y/n): ")
        if response.lower() == 'y':
            rollback(backup_file)
            return False
    
    print("=" * 50)
    print("🎉 Migration completed successfully!")
    print("\nNext steps:")
    print("1. Start the server: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("2. Test endpoints: http://localhost:8000/health")
    print("3. Check performance: http://localhost:8000/performance/status")
    print("4. View API docs: http://localhost:8000/docs")
    
    if backup_file:
        print(f"\nBackup available at: {backup_file}")
        print("If issues occur, run: cp {backup_file} main.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)