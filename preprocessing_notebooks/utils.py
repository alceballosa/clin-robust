import os
import re

ILLEGAL_CHARS = {'"': "&quot;", "'": "&apos;", "<": "&lt;", "<": "&gt;", "&": "&amp;"}

ILLEGAL_AND = {"&": "&amp;"}


def remove_text_format(text):
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_xml_illegal_chars(xml_src_path, xml_tar_path=None, escaped_patterns=[]):
    if xml_tar_path is None:
        xml_tar_path = xml_src_path

    with open(xml_src_path, "r") as f:
        xml_lines = f.readlines()

    for i in range(len(xml_lines)):
        escaped_texts = []
        for pattern in escaped_patterns:
            try:
                pattern = re.compile(pattern)
                texts = re.findall(pattern, xml_lines[i])
            except:
                print(f"Error: {pattern}")
                raise
            escaped_texts.extend(texts)

        for j, text in enumerate(escaped_texts):
            xml_lines[i] = xml_lines[i].replace(text, f"ESCAPED_TEXT{j}")

        for char, escaped_char in ILLEGAL_CHARS.items():
            xml_lines[i] = xml_lines[i].replace(char, escaped_char)

        for j, text in enumerate(escaped_texts):
            xml_lines[i] = xml_lines[i].replace(f"ESCAPED_TEXT{j}", text)

    if not os.path.exists(os.path.dirname(xml_tar_path)):
        os.makedirs(os.path.dirname(xml_tar_path))

    with open(xml_tar_path, "w") as f:
        f.writelines(xml_lines)


def preprocess_xml_illegal_and(xml_src_path, xml_tar_path=None, escaped_patterns=[]):
    if xml_tar_path is None:
        xml_tar_path = xml_src_path

    with open(xml_src_path, "r") as f:
        xml_lines = f.readlines()

    for i in range(len(xml_lines)):
        escaped_texts = []
        for pattern in escaped_patterns:
            try:
                pattern = re.compile(pattern)
                texts = re.findall(pattern, xml_lines[i])
            except:
                print(f"Error: {pattern}")
                raise
            escaped_texts.extend(texts)

        for j, text in enumerate(escaped_texts):
            xml_lines[i] = xml_lines[i].replace(text, f"ESCAPED_TEXT{j}")

        for char, escaped_char in ILLEGAL_AND.items():
            xml_lines[i] = xml_lines[i].replace(char, escaped_char)

        for j, text in enumerate(escaped_texts):
            xml_lines[i] = xml_lines[i].replace(f"ESCAPED_TEXT{j}", text)

    if not os.path.exists(os.path.dirname(xml_tar_path)):
        os.makedirs(os.path.dirname(xml_tar_path))

    with open(xml_tar_path, "w") as f:
        f.writelines(xml_lines)
