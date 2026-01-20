"""
Report Generator - HTML报告生成器

生成包含所有可视化图表的HTML报告
"""

import os
from typing import Dict, List, Any
from datetime import datetime


class ReportGenerator:
    """
    HTML报告生成器

    生成包含所有可视化图表的HTML报告
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化报告生成器

        Args:
            config: 配置字典
        """
        self.config = config
        self.save_dir = config.get("save_dir", "./figures")
        self.model_name = config.get("model_name", "model")
        self.report_name = config.get("report_name", "analysis_report")

        # 报告内容
        self.title = f"{self.model_name} - Outlier Analysis Report"
        self.sections = []

    def add_section(self, title: str, content: str, images: List[str] = None):
        """
        添加报告章节

        Args:
            title: 章节标题
            content: 章节内容（HTML格式）
            images: 图片路径列表
        """
        section = {"title": title, "content": content, "images": images or []}
        self.sections.append(section)

    def generate(self) -> str:
        """
        生成HTML报告

        Returns:
            保存的HTML文件路径
        """
        html_content = self._generate_html()

        # 保存HTML文件
        html_path = os.path.join(self.save_dir, f"{self.report_name}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    def _generate_html(self) -> str:
        """
        生成HTML内容

        Returns:
            HTML字符串
        """
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header .meta {{
            margin-top: 10px;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-container {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .image-caption {{
            padding: 10px;
            background: #f9f9f9;
            font-size: 0.9em;
            color: #666;
            text-align: center;
        }}
        .toc {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .toc h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .toc ul {{
            list-style: none;
            padding-left: 0;
        }}
        .toc li {{
            padding: 5px 0;
        }}
        .toc a {{
            color: #667eea;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .stats-table th, .stats-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .stats-table th {{
            background-color: #667eea;
            color: white;
        }}
        .stats-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <div class="meta">
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
    
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
"""

        # 添加目录
        for i, section in enumerate(self.sections):
            html += (
                f'            <li><a href="#section-{i}">{section["title"]}</a></li>\n'
            )

        html += """        </ul>
    </div>
"""

        # 添加章节
        for i, section in enumerate(self.sections):
            html += f"""
    <div class="section" id="section-{i}">
        <h2>{section["title"]}</h2>
        {section["content"]}
"""

            # 添加图片
            if section["images"]:
                html += '        <div class="image-grid">\n'
                for img_path in section["images"]:
                    img_name = os.path.basename(img_path)
                    html += f"""
            <div class="image-container">
                <img src="{img_name}" alt="{img_name}">
                <div class="image-caption">{img_name}</div>
            </div>
"""
                html += "        </div>\n"

            html += "    </div>\n"

        html += (
            """
    <div class="section">
        <h2>Configuration</h2>
        <p>Analysis configuration used for this report:</p>
"""
            + self._format_config()
            + """
    </div>
    
    <footer style="text-align: center; margin-top: 30px; color: #666;">
        <p>Generated by Quant Smooth Ana - Model Outlier Visualization Tool</p>
    </footer>
</body>
</html>
"""
        )

        return html

    def _format_config(self) -> str:
        """
        格式化配置信息

        Returns:
            HTML格式的配置信息
        """
        html = '<table class="stats-table">\n'
        html += "<tr><th>Parameter</th><th>Value</th></tr>\n"

        for key, value in self.config.items():
            if key == "save_dir":
                continue  # 跳过保存目录
            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"

        html += "</table>\n"
        return html

    def add_summary(self, stats: Dict[str, Any]):
        """
        添加统计摘要

        Args:
            stats: 统计数据字典
        """
        content = "<p>Summary statistics:</p>\n"
        content += '<table class="stats-table">\n'
        content += "<tr><th>Metric</th><th>Value</th></tr>\n"

        for key, value in stats.items():
            if isinstance(value, (int, float)):
                content += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>\n"
            else:
                content += f"<tr><td>{key}</td><td>{value}</td></tr>\n"

        content += "</table>\n"

        self.add_section("Summary", content)
