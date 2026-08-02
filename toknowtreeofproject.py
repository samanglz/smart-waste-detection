import os
import sys
from pathlib import Path

# --------------------------------------------
# پیکربندی اسکنر
# --------------------------------------------
ROOT_DIR = Path.cwd()
OUTPUT_FILE = "project_structure_report.txt"

# پوشه‌هایی که نمی‌خواهیم اسکن کنیم (برای جلوگیری از حجم بالا)
SKIP_DIRS = {
    '.git', '__pycache__', 'venv', 'env', '.venv', 
    '.idea', '.vscode', 'node_modules', 'dist', 'build', 
    'logs', 'checkpoints', 'weights', '.pytest_cache',
    'data'  # دیتاست شما حاوی فایل‌های سنگین تصویر است، فعلاً رد می‌شود
}

# پسوندهایی که محتوایشان را نمایش می‌دهیم
TEXT_EXTS = {
    '.py', '.yaml', '.yml', '.json', '.txt', '.md', 
    '.cfg', '.ini', '.toml', '.sh', '.bat', '.xml', 
    '.html', '.css', '.js', '.conf', '.dockerfile'
}

MAX_LINES = 200  # حداکثر خط برای هر فایل (تا خروجی بیش از حد بزرگ نشود)

# --------------------------------------------
# توابع کمکی
# --------------------------------------------

def should_skip_dir(dir_name: str) -> bool:
    return dir_name in SKIP_DIRS

def is_text_file(file_path: Path) -> bool:
    # فایل‌های بدون پسوند مثل Dockerfile را هم بررسی می‌کنیم
    if file_path.suffix.lower() in TEXT_EXTS:
        return True
    # بررسی نام فایل‌های خاص
    if file_path.name.upper() in {'DOCKERFILE', 'MAKEFILE'}:
        return True
    return False

def get_tree_lines(start_path: Path, prefix: str = "", is_last: bool = True) -> list:
    """
    تولید درخت پوشه‌ها به صورت متنی
    """
    lines = []
    if not start_path.exists():
        return lines

    # اگر پوشه اصلی است
    if start_path == ROOT_DIR:
        lines.append(f"{start_path.name}/")
        prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{start_path.name}/")

    # آپدیت پرافیکس برای فرزندان
    if is_last:
        new_prefix = prefix + "    "
    else:
        new_prefix = prefix + "│   "

    # دریافت آیتم‌های فرزند (مرتب‌سازی برای نظم)
    try:
        items = sorted(start_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return lines

    dirs = [item for item in items if item.is_dir() and not should_skip_dir(item.name)]
    files = [item for item in items if item.is_file()]

    # اضافه کردن زیرپوشه‌ها
    for idx, d in enumerate(dirs):
        is_last_item = (idx == len(dirs) - 1) and (len(files) == 0)
        lines.extend(get_tree_lines(d, new_prefix, is_last_item))

    # اضافه کردن فایل‌ها
    for idx, f in enumerate(files):
        connector = "└── " if idx == len(files) - 1 else "├── "
        lines.append(f"{new_prefix}{connector}{f.name}")

    return lines

def read_file_content(file_path: Path) -> str:
    """
    خواندن محتوای فایل با محدودیت خطوط
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        if total_lines > MAX_LINES:
            content = "".join(lines[:MAX_LINES])
            content += f"\n... (تعداد کل خطوط: {total_lines}، فقط {MAX_LINES} خط اول نمایش داده شده است)\n"
        else:
            content = "".join(lines)
        
        return content
    except Exception as e:
        return f"<خطا در خواندن فایل: {str(e)}>\n"

# --------------------------------------------
# اجرای اصلی
# --------------------------------------------

def main():
    print(f"در حال اسکن پروژه در مسیر: {ROOT_DIR}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        # 1. نوشتن درخت پروژه
        out.write("=" * 80 + "\n")
        out.write("                ساختار درختی پروژه (Project Tree)\n")
        out.write("=" * 80 + "\n\n")
        
        tree_lines = get_tree_lines(ROOT_DIR)
        out.write("\n".join(tree_lines))
        out.write("\n\n\n")

        # 2. نوشتن محتوای فایل‌های متنی مهم
        out.write("=" * 80 + "\n")
        out.write("              محتوای فایل‌های متنی (Text Files Content)\n")
        out.write("=" * 80 + "\n\n")

        # جمع‌آوری همه فایل‌های متنی (به جز آنهایی که در پوشه‌های SKIP قرار دارند)
        all_files = []
        for root, dirs, files in os.walk(ROOT_DIR):
            # حذف پوشه‌های اسکیپ شده از لیست walk تا وارد نشود
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            
            root_path = Path(root)
            for file_name in files:
                file_path = root_path / file_name
                if is_text_file(file_path):
                    all_files.append(file_path)

        # مرتب‌سازی بر اساس مسیر
        all_files.sort(key=lambda p: str(p))

        for file_path in all_files:
            # نمایش مسیر نسبی
            rel_path = file_path.relative_to(ROOT_DIR)
            out.write(f"\n--- FILE: {rel_path} ---\n")
            out.write(f"--- Path: {file_path} ---\n")
            content = read_file_content(file_path)
            out.write(content)
            out.write("\n" + "-" * 40 + "\n")

    print(f"اسکن کامل شد! گزارش در فایل '{OUTPUT_FILE}' ذخیره شد.")
    print(f"لطفاً محتوای این فایل را کپی کرده و برای من ارسال کنید.")

if __name__ == "__main__":
    main()