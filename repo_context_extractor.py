import os
import datetime
from pathlib import Path
from typing import Set, Dict, List, Optional
import pyperclip
import requests

# Configuration
class Config:
    FOLDER_EXCLUDE = {
        ".git", "__pycache__", "node_modules", ".venv", "archive", "deployment_scripts",
        "analysis-2024-09-15", "analysis-2024-08-30", "analysis-2024-08-29",
        "analysis-2024-08-09", "analysis-2024-08-05", "analysis-2024-07-26",
        "analysis-2024-07-19"
    }
    FILE_EXTENSION_EXCLUDE = {".exe", ".dll", ".so", ".pyc", ".png", ".jpg", ".jpeg", ".dbml", ".csv", ".hdf5", ".db"}
    SPECIFIC_FILES_MODE = False
    SPECIFIC_FILES = [
    "src/create_bq_data_table.py",
    "src/create_db_schema.py", 
    "src/cwsi_th1.py",
    "src/cwsi_th2_soybean.py",
    "src/dat_to_canopy_temp.py",
    "src/dat_to_csv.py",
    "src/et.py",
    "src/fuzz.py",
    "src/fuzz_with_visuals.py",
    "src/get_forecasts.py",
    "src/other_scheduling.py",
    "src/run_analysis.py",
    "src/swsi.py",
    "src/update_bigquery_tables.py",
    "sensor_mapping.yaml"
]
    CUSTOM_TAGS = [
        {"name": "instructions", "url": None},
        {"name": "error", "url": None},
        {
            "name": "custom_instructions",
            "url": "https://docs.google.com/document/d/1emAdwa-92zF8Jjx53qkMJX526rzIJ5ZSlxy0H2nrrzg/export?format=txt"
        }
    ]

class FileHandler:
    @staticmethod
    def read_file_content(file_path: Path) -> str:
        try:
            content = file_path.read_text(encoding="utf-8")
            if file_path.name == ".env":
                return FileHandler._obfuscate_env_file(content)
            return content
        except UnicodeDecodeError:
            return "Binary or non-UTF-8 content not displayed"
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def _obfuscate_env_file(content: str) -> str:
        return "\n".join(
            f"{line.split('=')[0].strip()}=********" if "=" in line else line
            for line in content.splitlines()
        )

class GoogleDocsHandler:
    @staticmethod
    def fetch_content(url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"<!-- Failed to fetch content: {e} -->"

class ContextBuilder:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def build(self) -> str:
        context_parts = [
            "<context>",
            f"    <timestamp>{self.timestamp}</timestamp>",
            self._build_custom_tags(),
            self._build_repository_structure(),
            self._build_additional_info(),
            "</context>"
        ]
        return "\n".join(context_parts)

    def _build_custom_tags(self) -> str:
        tags = []
        for tag in Config.CUSTOM_TAGS:
            name = tag.get("name", "")
            content = (
                GoogleDocsHandler.fetch_content(tag["url"])
                if tag.get("url")
                else f"<!-- Add your {name} here -->"
            )
            tags.append(f"    <{name}>{content}</{name}>")
        return "\n".join(tags)

    def _build_repository_structure(self) -> str:
        if Config.SPECIFIC_FILES_MODE:
            return self._build_specific_files_structure()
        return self._build_full_repository_structure()

    def _build_specific_files_structure(self) -> str:
        structure_parts = ["    <repository_structure>",
                         f"        <timestamp>{self.timestamp}</timestamp>"]
        
        for file_name in Config.SPECIFIC_FILES:
            file_path = self.root_path / file_name
            if file_path.exists():
                structure_parts.append(self._create_file_element(file_path))
        
        structure_parts.append("    </repository_structure>")
        return "\n".join(structure_parts)

    def _build_full_repository_structure(self) -> str:
        structure_parts = ["    <repository_structure>",
                         f"        <timestamp>{self.timestamp}</timestamp>"]
        
        for path in self.root_path.rglob("*"):
            if self._should_process_path(path):
                if path.is_file():
                    structure_parts.append(self._create_file_element(path))
        
        structure_parts.append("    </repository_structure>")
        return "\n".join(structure_parts)

    def _should_process_path(self, path: Path) -> bool:
        return not (
            any(part in Config.FOLDER_EXCLUDE for part in path.parts) or
            path.suffix in Config.FILE_EXTENSION_EXCLUDE or
            path.name == Path(__file__).name
        )

    def _create_file_element(self, file_path: Path) -> str:
        relative_path = file_path.relative_to(self.root_path)
        return "\n".join([
            "    <file>",
            f"        <name>{file_path.name}</name>",
            f"        <path>{relative_path}</path>",
            f"        <content><![CDATA[{FileHandler.read_file_content(file_path)}]]></content>",
            "    </file>"
        ])

    def _build_additional_info(self) -> str:
        info = []
        if Config.SPECIFIC_FILES_MODE:
            info.append(f"Specific files mode is enabled. Only including: {', '.join(Config.SPECIFIC_FILES)}")
        else:
            info.extend([
                f"Excluding directories: {', '.join(Config.FOLDER_EXCLUDE)}",
                f"Excluding extensions: {', '.join(Config.FILE_EXTENSION_EXCLUDE)}",
                "Sensitive information in .env files has been obfuscated",
                f"Current script ({Path(__file__).name}) is excluded"
            ])
        return f"    <additional_information>{' '.join(info)}</additional_information>"

def main():
    root_path = Path.cwd()
    context = ContextBuilder(root_path).build()
    pyperclip.copy(context)
    print("Repository context has been copied to the clipboard.")

if __name__ == "__main__":
    main()