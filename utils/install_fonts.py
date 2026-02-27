"""
Script to copy fonts from a source directory to matplotlib's font library in the current conda environment.
"""

import os
import shutil
import sys
import matplotlib
from pathlib import Path


def get_matplotlib_font_dir():
    """
    Get the matplotlib font directory for the current conda environment.
    """
    try:
        # Get matplotlib data path
        mpl_data_path = matplotlib.get_data_path()
        font_dir = os.path.join(mpl_data_path, 'fonts', 'ttf')
        return font_dir
    except Exception as e:
        print(f"Error getting matplotlib font directory: {e}")
        return None


def copy_fonts(source_dir, target_dir=None):
    """
    Copy font files from source directory to matplotlib font directory.

    Args:
        source_dir (str): Path to the source fonts directory
        target_dir (str, optional): Path to the target matplotlib font directory.
                                    If None, uses the default matplotlib font directory.
    """
    # Validate source directory
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return False

    if not os.path.isdir(source_dir):
        print(f"Error: '{source_dir}' is not a directory.")
        return False

    # Get target directory
    if target_dir is None:
        target_dir = get_matplotlib_font_dir()
        if target_dir is None:
            return False

    # Validate target directory
    if not os.path.exists(target_dir):
        print(f"Error: Target directory '{target_dir}' does not exist.")
        return False

    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print()

    # Supported font file extensions
    font_extensions = ['.ttf', '.otf', '.TTF', '.OTF']

    # Copy font files
    copied_count = 0
    skipped_count = 0

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if file is a font file
        if any(filename.endswith(ext) for ext in font_extensions):
            target_path = os.path.join(target_dir, filename)

            # Skip if font already exists
            if os.path.exists(target_path):
                print(f"Skipped (already exists): {filename}")
                skipped_count += 1
            else:
                try:
                    shutil.copy2(file_path, target_path)
                    print(f"Copied: {filename}")
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {filename}: {e}")

    print()
    print(f"Summary: {copied_count} font(s) copied, {skipped_count} font(s) skipped.")
    print()
    print("Note: You may need to clear matplotlib's font cache after installing new fonts.")
    print("To clear the cache, delete the matplotlib cache directory:")
    print("  - Linux/macOS: ~/.cache/matplotlib")
    print("  - Windows: %USERPROFILE%\\.matplotlib")

    return True


def main():
    """
    Main function to run the font installation script.
    """
    # Default source directory (utils/fonts folder)
    script_dir = Path(__file__).parent
    default_source_dir = os.path.join(script_dir, 'fonts')

    # Parse command line arguments
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    else:
        # Check if default fonts directory exists
        if os.path.exists(default_source_dir) and os.path.isdir(default_source_dir):
            source_dir = default_source_dir
        else:
            print("Usage: python install_fonts.py [source_directory]")
            print()
            print("Arguments:")
            print("  source_directory  Path to the directory containing font files")
            print()
            print("Examples:")
            print(f"  python install_fonts.py {default_source_dir}")
            print("  python install_fonts.py /path/to/your/fonts")
            print()
            print("If no source directory is provided, the script will look for a 'fonts' folder in the utils directory.")
            return

    # Copy fonts
    success = copy_fonts(source_dir)

    if success:
        print()
        print("Font installation completed successfully!")
    else:
        print()
        print("Font installation failed.")


if __name__ == "__main__":
    main()
