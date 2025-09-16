"""
Visual Content Validator
========================

Comprehensive validation of figures, tables, and other visual elements
including accessibility compliance and data visualization best practices.
"""

import re
import json
import base64
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import tempfile
import subprocess

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ColorAnalysis:
    """Analysis of color usage in visual content."""
    total_colors: int
    dominant_colors: List[str]
    colorblind_friendly: bool
    contrast_ratio: float
    accessibility_score: float


@dataclass
class TableAnalysis:
    """Analysis of table structure and content."""
    row_count: int
    column_count: int
    has_headers: bool
    has_captions: bool
    alignment_consistency: float
    readability_score: float
    data_density: float


@dataclass
class FigureAnalysis:
    """Analysis of figure quality and standards."""
    width_pixels: int
    height_pixels: int
    resolution_dpi: int
    file_format: str
    file_size_kb: float
    has_caption: bool
    caption_quality_score: float
    accessibility_features: List[str]
    visual_clarity_score: float


@dataclass
class VisualContentIssue:
    """Represents an issue found in visual content."""
    severity: str  # critical, warning, info
    issue_type: str
    description: str
    element_id: str
    recommendations: List[str]
    auto_fixable: bool


@dataclass
class VisualValidationResult:
    """Complete visual content validation result."""
    figure_analyses: List[FigureAnalysis]
    table_analyses: List[TableAnalysis]
    color_analyses: List[ColorAnalysis]
    identified_issues: List[VisualContentIssue]
    accessibility_score: float
    overall_quality_score: float
    recommendations: List[str]
    validation_summary: str


class VisualContentValidator:
    """Advanced visual content validation system."""
    
    def __init__(self, universal_chat_fn):
        self.universal_chat = universal_chat_fn
        
        # Quality standards
        self.min_figure_dpi = 300
        self.max_file_size_mb = 5
        self.recommended_formats = ['pdf', 'eps', 'svg', 'png']
        
        # Accessibility standards
        self.min_contrast_ratio = 4.5
        self.colorblind_safe_palettes = {
            'viridis': ['#440154', '#31688e', '#35b779', '#fde725'],
            'plasma': ['#0d0887', '#7e03a8', '#cc4778', '#f89441', '#f0f921'],
            'cividis': ['#00224e', '#123570', '#3b496c', '#575d6d', '#707173']
        }
        
        # Table standards
        self.max_table_width = 120  # characters
        self.min_cell_padding = 2
    
    def validate_visual_content(self, paper_content: str, paper_dir: Path,
                              model: str, request_timeout: int = 1800) -> VisualValidationResult:
        """Comprehensive visual content validation."""
        print("🎨 Starting Visual Content Validation...")
        
        # Extract visual elements
        figures = self._extract_figures(paper_content, paper_dir)
        tables = self._extract_tables(paper_content)
        
        print(f"  Found {len(figures)} figures and {len(tables)} tables")
        
        # Analyze figures
        figure_analyses = []
        for fig_info in figures:
            analysis = self._analyze_figure(fig_info, paper_dir)
            if analysis:
                figure_analyses.append(analysis)
        
        # Analyze tables
        table_analyses = []
        for table_info in tables:
            analysis = self._analyze_table(table_info)
            table_analyses.append(analysis)
        
        # Analyze color usage
        color_analyses = self._analyze_color_usage(figure_analyses)
        
        # Identify issues
        identified_issues = self._identify_visual_issues(
            figure_analyses, table_analyses, color_analyses
        )
        print(f"  Identified {len(identified_issues)} visual issues")
        
        # Calculate accessibility score
        accessibility_score = self._calculate_accessibility_score(
            figure_analyses, table_analyses, color_analyses
        )
        
        # Calculate overall quality score
        overall_quality_score = self._calculate_quality_score(
            figure_analyses, table_analyses, identified_issues
        )
        
        # Generate recommendations
        recommendations = self._generate_visual_recommendations(
            figure_analyses, table_analyses, identified_issues, model, request_timeout
        )
        
        # Generate validation summary
        validation_summary = self._generate_validation_summary(
            len(figures), len(tables), len(identified_issues),
            accessibility_score, overall_quality_score
        )
        
        return VisualValidationResult(
            figure_analyses=figure_analyses,
            table_analyses=table_analyses,
            color_analyses=color_analyses,
            identified_issues=identified_issues,
            accessibility_score=accessibility_score,
            overall_quality_score=overall_quality_score,
            recommendations=recommendations,
            validation_summary=validation_summary
        )
    
    def _extract_figures(self, paper_content: str, paper_dir: Path) -> List[Dict[str, Any]]:
        """Extract figure information from LaTeX content."""
        figures = []
        
        # Find figure environments
        figure_pattern = r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}'
        figure_matches = re.findall(figure_pattern, paper_content, re.DOTALL)
        
        for i, figure_content in enumerate(figure_matches):
            fig_info = {
                'id': f'figure_{i+1}',
                'content': figure_content.strip(),
                'caption': '',
                'label': '',
                'files': []
            }
            
            # Extract caption
            caption_match = re.search(r'\\caption\{([^}]+)\}', figure_content)
            if caption_match:
                fig_info['caption'] = caption_match.group(1)
            
            # Extract label
            label_match = re.search(r'\\label\{([^}]+)\}', figure_content)
            if label_match:
                fig_info['label'] = label_match.group(1)
            
            # Extract included files
            includegraphics_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
            file_matches = re.findall(includegraphics_pattern, figure_content)
            
            for filename in file_matches:
                # Try common extensions if none specified
                if '.' not in filename:
                    for ext in ['pdf', 'png', 'jpg', 'jpeg', 'eps', 'svg']:
                        test_path = paper_dir / f"{filename}.{ext}"
                        if test_path.exists():
                            fig_info['files'].append(str(test_path))
                            break
                else:
                    file_path = paper_dir / filename
                    if file_path.exists():
                        fig_info['files'].append(str(file_path))
            
            figures.append(fig_info)
        
        return figures
    
    def _extract_tables(self, paper_content: str) -> List[Dict[str, Any]]:
        """Extract table information from LaTeX content."""
        tables = []
        
        # Find table environments
        table_pattern = r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}'
        table_matches = re.findall(table_pattern, paper_content, re.DOTALL)
        
        for i, table_content in enumerate(table_matches):
            table_info = {
                'id': f'table_{i+1}',
                'content': table_content.strip(),
                'caption': '',
                'label': '',
                'tabular_content': ''
            }
            
            # Extract caption
            caption_match = re.search(r'\\caption\{([^}]+)\}', table_content)
            if caption_match:
                table_info['caption'] = caption_match.group(1)
            
            # Extract label
            label_match = re.search(r'\\label\{([^}]+)\}', table_content)
            if label_match:
                table_info['label'] = label_match.group(1)
            
            # Extract tabular content
            tabular_pattern = r'\\begin\{tabular\}.*?\{([^}]+)\}(.*?)\\end\{tabular\}'
            tabular_match = re.search(tabular_pattern, table_content, re.DOTALL)
            if tabular_match:
                table_info['column_spec'] = tabular_match.group(1)
                table_info['tabular_content'] = tabular_match.group(2)
            
            tables.append(table_info)
        
        return tables
    
    def _analyze_figure(self, fig_info: Dict[str, Any], paper_dir: Path) -> Optional[FigureAnalysis]:
        """Analyze individual figure quality."""
        if not fig_info['files']:
            return None
        
        # Analyze first file (primary figure)
        file_path = Path(fig_info['files'][0])
        
        if not file_path.exists():
            return None
        
        # Get file info
        file_size_kb = file_path.stat().st_size / 1024
        file_format = file_path.suffix.lower().lstrip('.')
        
        # Analyze image properties if possible
        width_pixels = 0
        height_pixels = 0
        resolution_dpi = 0
        
        if HAS_PIL and file_format in ['png', 'jpg', 'jpeg']:
            try:
                with Image.open(file_path) as img:
                    width_pixels, height_pixels = img.size
                    if 'dpi' in img.info:
                        resolution_dpi = img.info['dpi'][0]
            except Exception:
                pass
        
        # Analyze caption quality
        caption_quality_score = self._analyze_caption_quality(fig_info['caption'])
        
        # Check accessibility features
        accessibility_features = self._check_figure_accessibility(fig_info)
        
        # Calculate visual clarity score
        visual_clarity_score = self._calculate_visual_clarity_score(
            width_pixels, height_pixels, resolution_dpi, file_format
        )
        
        return FigureAnalysis(
            width_pixels=width_pixels,
            height_pixels=height_pixels,
            resolution_dpi=resolution_dpi,
            file_format=file_format,
            file_size_kb=file_size_kb,
            has_caption=bool(fig_info['caption']),
            caption_quality_score=caption_quality_score,
            accessibility_features=accessibility_features,
            visual_clarity_score=visual_clarity_score
        )
    
    def _analyze_table(self, table_info: Dict[str, Any]) -> TableAnalysis:
        """Analyze table structure and quality."""
        tabular_content = table_info.get('tabular_content', '')
        
        # Count rows and columns
        rows = [row.strip() for row in tabular_content.split('\\\\') if row.strip()]
        row_count = len(rows)
        
        column_count = 0
        if rows:
            # Count columns from first row
            first_row = rows[0]
            column_count = len(first_row.split('&'))
        
        # Check for headers
        has_headers = any('\\hline' in tabular_content or 
                         '\\toprule' in tabular_content or
                         '\\midrule' in tabular_content)
        
        # Check for captions
        has_captions = bool(table_info.get('caption', ''))
        
        # Analyze alignment consistency
        alignment_consistency = self._analyze_table_alignment(table_info.get('column_spec', ''))
        
        # Calculate readability score
        readability_score = self._calculate_table_readability(tabular_content, row_count, column_count)
        
        # Calculate data density
        total_cells = row_count * column_count
        data_density = total_cells / max(1, len(tabular_content)) if tabular_content else 0
        
        return TableAnalysis(
            row_count=row_count,
            column_count=column_count,
            has_headers=has_headers,
            has_captions=has_captions,
            alignment_consistency=alignment_consistency,
            readability_score=readability_score,
            data_density=data_density
        )
    
    def _analyze_caption_quality(self, caption: str) -> float:
        """Analyze quality of figure/table caption."""
        if not caption:
            return 0.0
        
        score = 0.0
        max_score = 5.0
        
        # Length check (good captions are descriptive but not too long)
        word_count = len(caption.split())
        if 10 <= word_count <= 50:
            score += 1.0
        elif 5 <= word_count <= 80:
            score += 0.5
        
        # Check for descriptive content
        descriptive_words = ['shows', 'displays', 'illustrates', 'demonstrates', 
                           'compares', 'presents', 'depicts', 'contains']
        if any(word in caption.lower() for word in descriptive_words):
            score += 1.0
        
        # Check for methodology description
        method_words = ['using', 'with', 'by', 'through', 'via', 'based on']
        if any(word in caption.lower() for word in method_words):
            score += 1.0
        
        # Check for statistical information
        stats_pattern = r'(p[<>=]\s*[\d.]+|n\s*=\s*\d+|±|error bars?)'
        if re.search(stats_pattern, caption.lower()):
            score += 1.0
        
        # Check for proper formatting
        if caption[0].isupper() and caption.endswith('.'):
            score += 1.0
        
        return score / max_score
    
    def _check_figure_accessibility(self, fig_info: Dict[str, Any]) -> List[str]:
        """Check accessibility features of figure."""
        features = []
        
        # Check if has descriptive caption
        caption = fig_info.get('caption', '')
        if caption and len(caption.split()) >= 10:
            features.append('descriptive_caption')
        
        # Check if caption describes visual elements
        visual_terms = ['color', 'red', 'blue', 'green', 'line', 'bar', 'point', 'marker']
        if any(term in caption.lower() for term in visual_terms):
            features.append('visual_description')
        
        # Check for alternative text indicators
        if 'alt' in fig_info.get('content', '').lower():
            features.append('alt_text')
        
        return features
    
    def _calculate_visual_clarity_score(self, width: int, height: int, 
                                      dpi: int, file_format: str) -> float:
        """Calculate visual clarity score based on technical parameters."""
        score = 0.0
        max_score = 5.0
        
        # Resolution score
        if dpi >= self.min_figure_dpi:
            score += 2.0
        elif dpi >= 150:
            score += 1.0
        elif dpi >= 72:
            score += 0.5
        
        # Format score
        if file_format in ['pdf', 'eps', 'svg']:
            score += 2.0  # Vector formats
        elif file_format == 'png':
            score += 1.5  # High-quality raster
        elif file_format in ['jpg', 'jpeg']:
            score += 1.0  # Acceptable raster
        
        # Size appropriateness
        if width >= 1200 and height >= 800:
            score += 1.0
        elif width >= 800 and height >= 600:
            score += 0.5
        
        return score / max_score
    
    def _analyze_color_usage(self, figure_analyses: List[FigureAnalysis]) -> List[ColorAnalysis]:
        """Analyze color usage across figures."""
        # Placeholder implementation - would analyze actual color usage
        color_analyses = []
        
        for fig_analysis in figure_analyses:
            # Simplified analysis
            analysis = ColorAnalysis(
                total_colors=5,  # Placeholder
                dominant_colors=['#1f77b4', '#ff7f0e'],  # Placeholder
                colorblind_friendly=True,  # Placeholder
                contrast_ratio=4.8,  # Placeholder
                accessibility_score=0.8  # Placeholder
            )
            color_analyses.append(analysis)
        
        return color_analyses
    
    def _analyze_table_alignment(self, column_spec: str) -> float:
        """Analyze consistency of table column alignment."""
        if not column_spec:
            return 0.5
        
        alignments = re.findall(r'[lcr]', column_spec.lower())
        
        if not alignments:
            return 0.5
        
        # Check for consistent alignment
        unique_alignments = set(alignments)
        consistency = 1.0 - (len(unique_alignments) - 1) / len(alignments)
        
        return consistency
    
    def _calculate_table_readability(self, content: str, rows: int, cols: int) -> float:
        """Calculate table readability score."""
        if not content or rows == 0 or cols == 0:
            return 0.0
        
        score = 0.0
        max_score = 4.0
        
        # Size appropriateness
        total_cells = rows * cols
        if total_cells <= 50:
            score += 1.0
        elif total_cells <= 100:
            score += 0.5
        
        # Formatting indicators
        if '\\hline' in content or '\\toprule' in content:
            score += 1.0
        
        if '\\midrule' in content:
            score += 1.0
        
        # Content density
        avg_cell_content = len(content) / total_cells
        if 5 <= avg_cell_content <= 30:
            score += 1.0
        elif 2 <= avg_cell_content <= 50:
            score += 0.5
        
        return score / max_score
    
    def _identify_visual_issues(self, figure_analyses: List[FigureAnalysis],
                              table_analyses: List[TableAnalysis],
                              color_analyses: List[ColorAnalysis]) -> List[VisualContentIssue]:
        """Identify issues in visual content."""
        issues = []
        
        # Figure issues
        for i, fig in enumerate(figure_analyses):
            element_id = f"figure_{i+1}"
            
            # Resolution issues
            if fig.resolution_dpi > 0 and fig.resolution_dpi < self.min_figure_dpi:
                issues.append(VisualContentIssue(
                    severity="warning",
                    issue_type="resolution",
                    description=f"Figure resolution ({fig.resolution_dpi} DPI) below recommended {self.min_figure_dpi} DPI",
                    element_id=element_id,
                    recommendations=["Increase figure resolution for better print quality"],
                    auto_fixable=False
                ))
            
            # File size issues
            if fig.file_size_kb > self.max_file_size_mb * 1024:
                issues.append(VisualContentIssue(
                    severity="warning",
                    issue_type="file_size",
                    description=f"Figure file size ({fig.file_size_kb:.1f} KB) is very large",
                    element_id=element_id,
                    recommendations=["Consider compressing the figure or using a more efficient format"],
                    auto_fixable=False
                ))
            
            # Format issues
            if fig.file_format not in self.recommended_formats:
                issues.append(VisualContentIssue(
                    severity="info",
                    issue_type="format",
                    description=f"Figure format ({fig.file_format}) not in recommended formats",
                    element_id=element_id,
                    recommendations=["Consider using PDF, EPS, SVG, or PNG format"],
                    auto_fixable=False
                ))
            
            # Caption issues
            if not fig.has_caption:
                issues.append(VisualContentIssue(
                    severity="critical",
                    issue_type="caption",
                    description="Figure missing caption",
                    element_id=element_id,
                    recommendations=["Add descriptive caption explaining what the figure shows"],
                    auto_fixable=False
                ))
            elif fig.caption_quality_score < 0.5:
                issues.append(VisualContentIssue(
                    severity="warning",
                    issue_type="caption_quality",
                    description="Figure caption could be more descriptive",
                    element_id=element_id,
                    recommendations=["Improve caption with more detail about methodology and interpretation"],
                    auto_fixable=False
                ))
            
            # Accessibility issues
            if 'descriptive_caption' not in fig.accessibility_features:
                issues.append(VisualContentIssue(
                    severity="warning",
                    issue_type="accessibility",
                    description="Figure may not be accessible to visually impaired readers",
                    element_id=element_id,
                    recommendations=["Add more descriptive text explaining visual elements"],
                    auto_fixable=False
                ))
        
        # Table issues
        for i, table in enumerate(table_analyses):
            element_id = f"table_{i+1}"
            
            # Caption issues
            if not table.has_captions:
                issues.append(VisualContentIssue(
                    severity="critical",
                    issue_type="caption",
                    description="Table missing caption",
                    element_id=element_id,
                    recommendations=["Add caption explaining table contents and methodology"],
                    auto_fixable=False
                ))
            
            # Header issues
            if not table.has_headers:
                issues.append(VisualContentIssue(
                    severity="warning",
                    issue_type="structure",
                    description="Table appears to be missing headers",
                    element_id=element_id,
                    recommendations=["Add clear column headers with \\hline or \\toprule"],
                    auto_fixable=False
                ))
            
            # Size issues
            if table.row_count * table.column_count > 100:
                issues.append(VisualContentIssue(
                    severity="info",
                    issue_type="size",
                    description="Table is very large and may be hard to read",
                    element_id=element_id,
                    recommendations=["Consider splitting into multiple tables or moving to appendix"],
                    auto_fixable=False
                ))
            
            # Readability issues
            if table.readability_score < 0.5:
                issues.append(VisualContentIssue(
                    severity="warning",
                    issue_type="readability",
                    description="Table formatting could be improved for readability",
                    element_id=element_id,
                    recommendations=["Improve table formatting with better alignment and spacing"],
                    auto_fixable=False
                ))
        
        # Color issues
        for i, color in enumerate(color_analyses):
            if not color.colorblind_friendly:
                issues.append(VisualContentIssue(
                    severity="warning",
                    issue_type="accessibility",
                    description=f"Figure {i+1} may not be colorblind-friendly",
                    element_id=f"figure_{i+1}",
                    recommendations=["Use colorblind-safe palette or add patterns/shapes"],
                    auto_fixable=False
                ))
            
            if color.contrast_ratio < self.min_contrast_ratio:
                issues.append(VisualContentIssue(
                    severity="warning",
                    issue_type="contrast",
                    description=f"Figure {i+1} has low color contrast ratio",
                    element_id=f"figure_{i+1}",
                    recommendations=["Increase color contrast for better readability"],
                    auto_fixable=False
                ))
        
        return issues
    
    def _calculate_accessibility_score(self, figure_analyses: List[FigureAnalysis],
                                     table_analyses: List[TableAnalysis],
                                     color_analyses: List[ColorAnalysis]) -> float:
        """Calculate overall accessibility score."""
        if not (figure_analyses or table_analyses):
            return 1.0
        
        total_elements = len(figure_analyses) + len(table_analyses)
        accessibility_points = 0.0
        
        # Figure accessibility
        for fig in figure_analyses:
            points = 0.0
            if fig.has_caption:
                points += 0.3
            if fig.caption_quality_score >= 0.5:
                points += 0.3
            if len(fig.accessibility_features) >= 2:
                points += 0.4
            accessibility_points += points
        
        # Table accessibility
        for table in table_analyses:
            points = 0.0
            if table.has_captions:
                points += 0.4
            if table.has_headers:
                points += 0.3
            if table.readability_score >= 0.5:
                points += 0.3
            accessibility_points += points
        
        # Color accessibility
        colorblind_friendly_count = sum(1 for color in color_analyses if color.colorblind_friendly)
        if color_analyses:
            color_accessibility = colorblind_friendly_count / len(color_analyses)
            accessibility_points += color_accessibility * len(figure_analyses)
        
        return accessibility_points / total_elements if total_elements > 0 else 1.0
    
    def _calculate_quality_score(self, figure_analyses: List[FigureAnalysis],
                               table_analyses: List[TableAnalysis],
                               issues: List[VisualContentIssue]) -> float:
        """Calculate overall visual quality score."""
        if not (figure_analyses or table_analyses):
            return 1.0
        
        # Base score from technical quality
        figure_scores = [fig.visual_clarity_score for fig in figure_analyses]
        table_scores = [table.readability_score for table in table_analyses]
        
        all_scores = figure_scores + table_scores
        base_score = sum(all_scores) / len(all_scores) if all_scores else 1.0
        
        # Penalty for issues
        critical_issues = [i for i in issues if i.severity == 'critical']
        warning_issues = [i for i in issues if i.severity == 'warning']
        
        penalty = (len(critical_issues) * 0.2) + (len(warning_issues) * 0.1)
        
        final_score = max(0.0, base_score - penalty)
        return min(1.0, final_score)
    
    def _generate_visual_recommendations(self, figure_analyses: List[FigureAnalysis],
                                       table_analyses: List[TableAnalysis],
                                       issues: List[VisualContentIssue],
                                       model: str, request_timeout: int) -> List[str]:
        """Generate visual content improvement recommendations."""
        recommendations = []
        
        # Issue-based recommendations
        critical_issues = [i for i in issues if i.severity == 'critical']
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical visual issues immediately")
        
        warning_issues = [i for i in issues if i.severity == 'warning']
        if warning_issues:
            recommendations.append(f"Consider fixing {len(warning_issues)} visual quality warnings")
        
        # Figure-specific recommendations
        low_res_figures = [f for f in figure_analyses if f.resolution_dpi > 0 and f.resolution_dpi < self.min_figure_dpi]
        if low_res_figures:
            recommendations.append(f"Increase resolution for {len(low_res_figures)} figures to at least {self.min_figure_dpi} DPI")
        
        # Caption quality recommendations
        poor_captions = [f for f in figure_analyses if f.caption_quality_score < 0.5]
        if poor_captions:
            recommendations.append("Improve figure captions with more descriptive and methodological details")
        
        # Table recommendations
        large_tables = [t for t in table_analyses if t.row_count * t.column_count > 50]
        if large_tables:
            recommendations.append("Consider simplifying or splitting large tables for better readability")
        
        # Accessibility recommendations
        accessibility_issues = [i for i in issues if i.issue_type == 'accessibility']
        if accessibility_issues:
            recommendations.append("Improve accessibility with better color choices and more descriptive text")
        
        # General quality recommendations
        if not recommendations:
            recommendations.append("Visual content meets quality standards - consider minor formatting improvements")
        
        return recommendations
    
    def _generate_validation_summary(self, figure_count: int, table_count: int,
                                   issue_count: int, accessibility_score: float,
                                   quality_score: float) -> str:
        """Generate visual validation summary."""
        summary = []
        
        summary.append(f"Visual Content Validation Summary:")
        summary.append(f"- Figures analyzed: {figure_count}")
        summary.append(f"- Tables analyzed: {table_count}")
        summary.append(f"- Issues identified: {issue_count}")
        summary.append(f"- Accessibility score: {accessibility_score:.2f}")
        summary.append(f"- Overall quality score: {quality_score:.2f}")
        
        # Quality assessment
        if quality_score >= 0.8 and accessibility_score >= 0.8:
            summary.append("- Overall assessment: EXCELLENT")
        elif quality_score >= 0.6 and accessibility_score >= 0.6:
            summary.append("- Overall assessment: GOOD")
        elif quality_score >= 0.4 or accessibility_score >= 0.4:
            summary.append("- Overall assessment: NEEDS IMPROVEMENT")
        else:
            summary.append("- Overall assessment: POOR - REQUIRES ATTENTION")
        
        return "\n".join(summary)
    
    def generate_visual_report(self, validation: VisualValidationResult) -> str:
        """Generate comprehensive visual content validation report."""
        report = []
        report.append("=" * 80)
        report.append("VISUAL CONTENT VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("VALIDATION SUMMARY:")
        for line in validation.validation_summary.split('\n'):
            report.append(f"  {line}")
        report.append("")
        
        # Issues
        if validation.identified_issues:
            report.append("IDENTIFIED ISSUES:")
            critical_issues = [i for i in validation.identified_issues if i.severity == 'critical']
            warning_issues = [i for i in validation.identified_issues if i.severity == 'warning']
            info_issues = [i for i in validation.identified_issues if i.severity == 'info']
            
            if critical_issues:
                report.append("  CRITICAL:")
                for issue in critical_issues:
                    report.append(f"    • {issue.element_id}: {issue.description}")
                    for rec in issue.recommendations:
                        report.append(f"      → {rec}")
            
            if warning_issues:
                report.append("  WARNING:")
                for issue in warning_issues:
                    report.append(f"    • {issue.element_id}: {issue.description}")
            
            if info_issues:
                report.append("  INFO:")
                for issue in info_issues:
                    report.append(f"    • {issue.element_id}: {issue.description}")
            
            report.append("")
        
        # Figure analysis
        if validation.figure_analyses:
            report.append("FIGURE ANALYSIS:")
            for i, fig in enumerate(validation.figure_analyses, 1):
                report.append(f"  Figure {i}:")
                report.append(f"    Resolution: {fig.resolution_dpi} DPI")
                report.append(f"    Format: {fig.file_format}")
                report.append(f"    Size: {fig.file_size_kb:.1f} KB")
                report.append(f"    Caption Quality: {fig.caption_quality_score:.2f}")
                report.append(f"    Visual Clarity: {fig.visual_clarity_score:.2f}")
            report.append("")
        
        # Table analysis
        if validation.table_analyses:
            report.append("TABLE ANALYSIS:")
            for i, table in enumerate(validation.table_analyses, 1):
                report.append(f"  Table {i}:")
                report.append(f"    Dimensions: {table.row_count}×{table.column_count}")
                report.append(f"    Has Headers: {table.has_headers}")
                report.append(f"    Has Caption: {table.has_captions}")
                report.append(f"    Readability: {table.readability_score:.2f}")
            report.append("")
        
        # Recommendations
        if validation.recommendations:
            report.append("RECOMMENDATIONS:")
            for rec in validation.recommendations:
                report.append(f"  • {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
