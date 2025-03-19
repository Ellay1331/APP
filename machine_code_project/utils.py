# -*- coding: utf-8 -*-
import os, re, logging

def translate_to_c(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Ошибка чтения файла {file_path}: {e}")
        content = ""
    if ext in (".c", ".h"):
        return content
    return f"/* Translated to C from {ext} */\n" + content

def remove_c_comments(code):
    pattern = r'//.*?$|/\*.*?\*/'
    return re.sub(pattern, '', code, flags=re.MULTILINE | re.DOTALL)

def remove_python_comments(code):
    return re.sub(r'#.*', '', code)

def remove_js_comments(code):
    pattern = r'//.*?$|/\*.*?\*/'
    return re.sub(pattern, '', code, flags=re.MULTILINE | re.DOTALL)
