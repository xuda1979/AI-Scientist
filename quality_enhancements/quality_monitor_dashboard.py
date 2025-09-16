"""
Quality Monitor Dashboard
=========================

Real-time quality monitoring system with trend analysis,
quality metrics tracking, and automated reporting.
"""

import json
import sqlite3
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import hashlib

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class QualityMetric:
    """Individual quality metric measurement."""
    metric_name: str
    value: float
    timestamp: datetime
    paper_id: str
    metric_type: str  # overall, technical, style, content
    details: Dict[str, Any]


@dataclass
class QualityTrend:
    """Quality trend analysis."""
    metric_name: str
    trend_direction: str  # improving, declining, stable
    change_rate: float
    confidence: float
    period_days: int
    recent_values: List[float]


@dataclass
class QualityAlert:
    """Quality alert for concerning trends."""
    alert_type: str  # threshold, trend, anomaly
    severity: str  # critical, warning, info
    message: str
    metric_name: str
    current_value: float
    threshold_value: Optional[float]
    timestamp: datetime


@dataclass
class PaperQualitySnapshot:
    """Complete quality snapshot for a paper."""
    paper_id: str
    timestamp: datetime
    overall_score: float
    technical_score: float
    style_score: float
    content_score: float
    individual_metrics: Dict[str, float]
    improvement_suggestions: List[str]
    quality_grade: str  # A, B, C, D, F


@dataclass
class QualityDashboardData:
    """Complete dashboard data."""
    current_metrics: Dict[str, QualityMetric]
    quality_trends: List[QualityTrend]
    active_alerts: List[QualityAlert]
    recent_papers: List[PaperQualitySnapshot]
    performance_summary: Dict[str, Any]
    trend_charts: Dict[str, str]  # metric_name -> chart_path


class QualityMonitorDashboard:
    """Real-time quality monitoring dashboard."""
    
    def __init__(self, db_path: Optional[Path] = None, 
                 dashboard_dir: Optional[Path] = None):
        
        # Database setup
        self.db_path = db_path or Path.home() / ".sciresearch" / "quality_monitor.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dashboard output directory
        self.dashboard_dir = dashboard_dir or Path.home() / ".sciresearch" / "dashboard"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Quality thresholds for alerts
        self.quality_thresholds = {
            'overall_score': {'critical': 0.4, 'warning': 0.6},
            'technical_score': {'critical': 0.3, 'warning': 0.5},
            'style_score': {'critical': 0.4, 'warning': 0.6},
            'content_score': {'critical': 0.3, 'warning': 0.5},
            'readability_score': {'critical': 30, 'warning': 40},
            'citation_density': {'critical': 5, 'warning': 10},
            'methodology_score': {'critical': 0.4, 'warning': 0.6}
        }
        
        # Monitoring configuration
        self.monitoring_enabled = False
        self.monitoring_thread = None
        self.update_interval = 60  # seconds
        
        # Recent metrics cache
        self.metrics_cache = defaultdict(lambda: deque(maxlen=100))
        self.cache_lock = threading.Lock()
    
    def _init_database(self):
        """Initialize SQLite database for storing quality metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Quality metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    UNIQUE(paper_id, metric_name, timestamp)
                )
            ''')
            
            # Quality snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    technical_score REAL NOT NULL,
                    style_score REAL NOT NULL,
                    content_score REAL NOT NULL,
                    quality_grade TEXT NOT NULL,
                    metrics_json TEXT,
                    suggestions_json TEXT
                )
            ''')
            
            # Quality alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL,
                    threshold_value REAL,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_paper_time ON quality_metrics(paper_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_time ON quality_snapshots(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_time ON quality_alerts(timestamp, resolved)')
            
            conn.commit()
    
    def record_quality_metrics(self, paper_id: str, metrics: Dict[str, Any]):
        """Record quality metrics for a paper."""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    value = metric_data.get('score', 0.0)
                    metric_type = metric_data.get('type', 'general')
                    details = json.dumps(metric_data.get('details', {}))
                else:
                    value = float(metric_data) if metric_data is not None else 0.0
                    metric_type = 'general'
                    details = '{}'
                
                # Insert metric
                cursor.execute('''
                    INSERT OR REPLACE INTO quality_metrics 
                    (paper_id, metric_name, metric_value, metric_type, timestamp, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (paper_id, metric_name, value, metric_type, timestamp, details))
                
                # Update cache
                with self.cache_lock:
                    metric = QualityMetric(
                        metric_name=metric_name,
                        value=value,
                        timestamp=datetime.fromisoformat(timestamp),
                        paper_id=paper_id,
                        metric_type=metric_type,
                        details=json.loads(details) if details else {}
                    )
                    self.metrics_cache[metric_name].append(metric)
            
            conn.commit()
        
        # Check for alerts
        self._check_quality_alerts(paper_id, metrics)
    
    def record_paper_snapshot(self, snapshot: PaperQualitySnapshot):
        """Record complete quality snapshot for a paper."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO quality_snapshots
                (paper_id, timestamp, overall_score, technical_score, 
                 style_score, content_score, quality_grade, metrics_json, suggestions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.paper_id,
                snapshot.timestamp.isoformat(),
                snapshot.overall_score,
                snapshot.technical_score,
                snapshot.style_score,
                snapshot.content_score,
                snapshot.quality_grade,
                json.dumps(snapshot.individual_metrics),
                json.dumps(snapshot.improvement_suggestions)
            ))
            
            conn.commit()
    
    def get_dashboard_data(self, hours_back: int = 24) -> QualityDashboardData:
        """Get complete dashboard data."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Get current metrics
        current_metrics = self._get_current_metrics()
        
        # Get quality trends
        quality_trends = self._analyze_quality_trends(days_back=7)
        
        # Get active alerts
        active_alerts = self._get_active_alerts()
        
        # Get recent paper snapshots
        recent_papers = self._get_recent_paper_snapshots(cutoff_time)
        
        # Generate performance summary
        performance_summary = self._generate_performance_summary(cutoff_time)
        
        # Generate trend charts
        trend_charts = self._generate_trend_charts(days_back=7)
        
        return QualityDashboardData(
            current_metrics=current_metrics,
            quality_trends=quality_trends,
            active_alerts=active_alerts,
            recent_papers=recent_papers,
            performance_summary=performance_summary,
            trend_charts=trend_charts
        )
    
    def _get_current_metrics(self) -> Dict[str, QualityMetric]:
        """Get latest metrics for each metric type."""
        current_metrics = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get latest metric for each metric name
            cursor.execute('''
                SELECT metric_name, metric_value, metric_type, timestamp, 
                       paper_id, details
                FROM quality_metrics
                WHERE timestamp = (
                    SELECT MAX(timestamp) 
                    FROM quality_metrics qm2 
                    WHERE qm2.metric_name = quality_metrics.metric_name
                )
                ORDER BY metric_name
            ''')
            
            for row in cursor.fetchall():
                metric_name, value, metric_type, timestamp, paper_id, details = row
                
                current_metrics[metric_name] = QualityMetric(
                    metric_name=metric_name,
                    value=value,
                    timestamp=datetime.fromisoformat(timestamp),
                    paper_id=paper_id,
                    metric_type=metric_type,
                    details=json.loads(details) if details else {}
                )
        
        return current_metrics
    
    def _analyze_quality_trends(self, days_back: int = 7) -> List[QualityTrend]:
        """Analyze quality trends over time."""
        trends = []
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get metrics for each metric name
            cursor.execute('''
                SELECT metric_name, metric_value, timestamp
                FROM quality_metrics
                WHERE timestamp > ?
                ORDER BY metric_name, timestamp
            ''', (cutoff_time.isoformat(),))
            
            # Group by metric name
            metrics_by_name = defaultdict(list)
            for metric_name, value, timestamp in cursor.fetchall():
                metrics_by_name[metric_name].append((value, datetime.fromisoformat(timestamp)))
        
        # Analyze trends for each metric
        for metric_name, values_times in metrics_by_name.items():
            if len(values_times) < 3:  # Need at least 3 points for trend
                continue
            
            values = [vt[0] for vt in values_times]
            times = [vt[1] for vt in values_times]
            
            trend = self._calculate_trend(values, times)
            if trend:
                trends.append(trend)
        
        return trends
    
    def _calculate_trend(self, values: List[float], times: List[datetime]) -> Optional[QualityTrend]:
        """Calculate trend for a series of values."""
        if len(values) < 3 or not HAS_NUMPY:
            return None
        
        try:
            # Convert times to numeric values (hours from first timestamp)
            time_numeric = [(t - times[0]).total_seconds() / 3600 for t in times]
            
            # Calculate linear regression
            coeffs = np.polyfit(time_numeric, values, 1)
            slope = coeffs[0]
            
            # Determine trend direction
            if abs(slope) < 0.001:  # Very small slope
                trend_direction = 'stable'
                change_rate = 0.0
            elif slope > 0:
                trend_direction = 'improving'
                change_rate = slope
            else:
                trend_direction = 'declining'
                change_rate = abs(slope)
            
            # Calculate confidence based on R-squared
            y_pred = np.polyval(coeffs, time_numeric)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            confidence = max(0.0, min(1.0, r_squared))
            
            return QualityTrend(
                metric_name=times[0].strftime('%Y-%m-%d'),  # Use first time as identifier
                trend_direction=trend_direction,
                change_rate=change_rate,
                confidence=confidence,
                period_days=len(values),
                recent_values=values[-5:]  # Last 5 values
            )
            
        except Exception:
            return None
    
    def _get_active_alerts(self) -> List[QualityAlert]:
        """Get active quality alerts."""
        alerts = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT alert_type, severity, message, metric_name,
                       current_value, threshold_value, timestamp
                FROM quality_alerts
                WHERE resolved = FALSE
                ORDER BY timestamp DESC
                LIMIT 50
            ''')
            
            for row in cursor.fetchall():
                alert_type, severity, message, metric_name, current_value, threshold_value, timestamp = row
                
                alerts.append(QualityAlert(
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=threshold_value,
                    timestamp=datetime.fromisoformat(timestamp)
                ))
        
        return alerts
    
    def _get_recent_paper_snapshots(self, cutoff_time: datetime) -> List[PaperQualitySnapshot]:
        """Get recent paper quality snapshots."""
        snapshots = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT paper_id, timestamp, overall_score, technical_score,
                       style_score, content_score, quality_grade,
                       metrics_json, suggestions_json
                FROM quality_snapshots
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (cutoff_time.isoformat(),))
            
            for row in cursor.fetchall():
                (paper_id, timestamp, overall_score, technical_score,
                 style_score, content_score, quality_grade, 
                 metrics_json, suggestions_json) = row
                
                snapshots.append(PaperQualitySnapshot(
                    paper_id=paper_id,
                    timestamp=datetime.fromisoformat(timestamp),
                    overall_score=overall_score,
                    technical_score=technical_score,
                    style_score=style_score,
                    content_score=content_score,
                    individual_metrics=json.loads(metrics_json) if metrics_json else {},
                    improvement_suggestions=json.loads(suggestions_json) if suggestions_json else [],
                    quality_grade=quality_grade
                ))
        
        return snapshots
    
    def _generate_performance_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        summary = {
            'total_papers': 0,
            'avg_overall_score': 0.0,
            'score_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
            'improvement_trend': 'stable',
            'top_issues': [],
            'processing_time': {
                'avg_seconds': 0.0,
                'min_seconds': 0.0,
                'max_seconds': 0.0
            }
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total papers processed
            cursor.execute('''
                SELECT COUNT(DISTINCT paper_id)
                FROM quality_snapshots
                WHERE timestamp > ?
            ''', (cutoff_time.isoformat(),))
            
            result = cursor.fetchone()
            if result:
                summary['total_papers'] = result[0]
            
            # Average scores
            cursor.execute('''
                SELECT AVG(overall_score)
                FROM quality_snapshots
                WHERE timestamp > ?
            ''', (cutoff_time.isoformat(),))
            
            result = cursor.fetchone()
            if result and result[0]:
                summary['avg_overall_score'] = result[0]
            
            # Grade distribution
            cursor.execute('''
                SELECT quality_grade, COUNT(*)
                FROM quality_snapshots
                WHERE timestamp > ?
                GROUP BY quality_grade
            ''', (cutoff_time.isoformat(),))
            
            for grade, count in cursor.fetchall():
                if grade in summary['score_distribution']:
                    summary['score_distribution'][grade] = count
            
            # Most common issues (simplified)
            cursor.execute('''
                SELECT metric_name, COUNT(*) as issue_count
                FROM quality_alerts
                WHERE timestamp > ? AND severity IN ('critical', 'warning')
                GROUP BY metric_name
                ORDER BY issue_count DESC
                LIMIT 5
            ''', (cutoff_time.isoformat(),))
            
            summary['top_issues'] = [
                {'metric': metric, 'count': count}
                for metric, count in cursor.fetchall()
            ]
        
        return summary
    
    def _generate_trend_charts(self, days_back: int = 7) -> Dict[str, str]:
        """Generate trend charts for key metrics."""
        if not HAS_MATPLOTLIB:
            return {}
        
        charts = {}
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        # Key metrics to chart
        key_metrics = ['overall_score', 'technical_score', 'style_score', 'content_score']
        
        for metric_name in key_metrics:
            chart_path = self._create_metric_chart(metric_name, cutoff_time)
            if chart_path:
                charts[metric_name] = str(chart_path)
        
        return charts
    
    def _create_metric_chart(self, metric_name: str, cutoff_time: datetime) -> Optional[Path]:
        """Create a chart for a specific metric."""
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            # Get data
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT metric_value, timestamp
                    FROM quality_metrics
                    WHERE metric_name = ? AND timestamp > ?
                    ORDER BY timestamp
                ''', (metric_name, cutoff_time.isoformat()))
                
                data = cursor.fetchall()
            
            if len(data) < 2:
                return None
            
            values = [row[0] for row in data]
            timestamps = [datetime.fromisoformat(row[1]) for row in data]
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(timestamps, values, marker='o', linewidth=2, markersize=4)
            ax.set_title(f'{metric_name.replace("_", " ").title()} Trend', fontsize=14)
            ax.set_xlabel('Time')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.xticks(rotation=45)
            
            # Add trend line if enough data
            if len(values) >= 3 and HAS_NUMPY:
                time_numeric = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
                coeffs = np.polyfit(time_numeric, values, 1)
                trend_line = np.polyval(coeffs, time_numeric)
                ax.plot(timestamps, trend_line, '--', alpha=0.7, color='red', label='Trend')
                ax.legend()
            
            plt.tight_layout()
            
            # Save chart
            chart_filename = f"{metric_name}_trend_{int(time.time())}.png"
            chart_path = self.dashboard_dir / "charts" / chart_filename
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            
            fig.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            return chart_path
            
        except Exception as e:
            print(f"⚠ Failed to create chart for {metric_name}: {e}")
            return None
    
    def _check_quality_alerts(self, paper_id: str, metrics: Dict[str, Any]):
        """Check for quality alerts based on current metrics."""
        alerts_to_add = []
        timestamp = datetime.now()
        
        for metric_name, metric_data in metrics.items():
            value = metric_data if isinstance(metric_data, (int, float)) else metric_data.get('score', 0)
            
            if metric_name not in self.quality_thresholds:
                continue
            
            thresholds = self.quality_thresholds[metric_name]
            
            # Check critical threshold
            if value < thresholds['critical']:
                alert = QualityAlert(
                    alert_type='threshold',
                    severity='critical',
                    message=f"{metric_name} ({value:.2f}) below critical threshold ({thresholds['critical']})",
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=thresholds['critical'],
                    timestamp=timestamp
                )
                alerts_to_add.append(alert)
            
            # Check warning threshold
            elif value < thresholds['warning']:
                alert = QualityAlert(
                    alert_type='threshold',
                    severity='warning',
                    message=f"{metric_name} ({value:.2f}) below warning threshold ({thresholds['warning']})",
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=thresholds['warning'],
                    timestamp=timestamp
                )
                alerts_to_add.append(alert)
        
        # Store alerts
        if alerts_to_add:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for alert in alerts_to_add:
                    cursor.execute('''
                        INSERT INTO quality_alerts
                        (alert_type, severity, message, metric_name,
                         current_value, threshold_value, timestamp, resolved)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.alert_type, alert.severity, alert.message,
                        alert.metric_name, alert.current_value, alert.threshold_value,
                        alert.timestamp.isoformat(), False
                    ))
                
                conn.commit()
    
    def start_monitoring(self):
        """Start real-time quality monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("📊 Quality monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time quality monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("📊 Quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Update dashboard data
                self._update_dashboard()
                
                # Clean old data
                self._cleanup_old_data()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"⚠ Monitoring error: {e}")
                time.sleep(60)  # Wait before retry
    
    def _update_dashboard(self):
        """Update dashboard with latest data."""
        # This would be called periodically to refresh dashboard
        dashboard_data = self.get_dashboard_data()
        
        # Generate HTML dashboard (simplified)
        self._generate_html_dashboard(dashboard_data)
    
    def _generate_html_dashboard(self, data: QualityDashboardData):
        """Generate HTML dashboard."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Paper Quality Dashboard</title>
            <meta http-equiv="refresh" content="60">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ background: #f5f5f5; padding: 15px; margin: 10px; border-radius: 5px; }}
                .alert-critical {{ background: #ffebee; border-left: 5px solid #f44336; }}
                .alert-warning {{ background: #fff3e0; border-left: 5px solid #ff9800; }}
                .score-excellent {{ color: #4caf50; }}
                .score-good {{ color: #8bc34a; }}
                .score-fair {{ color: #ff9800; }}
                .score-poor {{ color: #f44336; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Research Paper Quality Dashboard</h1>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Current Quality Metrics</h2>
            <div style="display: flex; flex-wrap: wrap;">
        """
        
        # Add current metrics
        for metric_name, metric in data.current_metrics.items():
            score_class = self._get_score_class(metric.value)
            html_content += f"""
                <div class="metric-card">
                    <h3>{metric_name.replace('_', ' ').title()}</h3>
                    <p class="{score_class}">{metric.value:.2f}</p>
                    <small>From: {metric.paper_id[:10]}...</small>
                </div>
            """
        
        html_content += "</div>"
        
        # Add active alerts
        if data.active_alerts:
            html_content += "<h2>Active Alerts</h2>"
            for alert in data.active_alerts[:10]:
                alert_class = f"alert-{alert.severity}"
                html_content += f"""
                    <div class="metric-card {alert_class}">
                        <strong>{alert.severity.upper()}:</strong> {alert.message}
                        <br><small>{alert.timestamp.strftime('%Y-%m-%d %H:%M')}</small>
                    </div>
                """
        
        # Add performance summary
        summary = data.performance_summary
        html_content += f"""
            <h2>Performance Summary</h2>
            <div class="metric-card">
                <p><strong>Total Papers:</strong> {summary['total_papers']}</p>
                <p><strong>Average Score:</strong> {summary['avg_overall_score']:.2f}</p>
                <p><strong>Grade Distribution:</strong></p>
                <ul>
                    <li>A: {summary['score_distribution']['A']}</li>
                    <li>B: {summary['score_distribution']['B']}</li>
                    <li>C: {summary['score_distribution']['C']}</li>
                    <li>D: {summary['score_distribution']['D']}</li>
                    <li>F: {summary['score_distribution']['F']}</li>
                </ul>
            </div>
        """
        
        # Add recent papers
        if data.recent_papers:
            html_content += """
                <h2>Recent Papers</h2>
                <table>
                    <tr>
                        <th>Paper ID</th>
                        <th>Overall Score</th>
                        <th>Grade</th>
                        <th>Timestamp</th>
                    </tr>
            """
            
            for paper in data.recent_papers[:10]:
                score_class = self._get_score_class(paper.overall_score)
                html_content += f"""
                    <tr>
                        <td>{paper.paper_id[:15]}...</td>
                        <td class="{score_class}">{paper.overall_score:.2f}</td>
                        <td>{paper.quality_grade}</td>
                        <td>{paper.timestamp.strftime('%m/%d %H:%M')}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
            </body>
        </html>
        """
        
        # Save HTML dashboard
        dashboard_file = self.dashboard_dir / "quality_dashboard.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score."""
        if score >= 0.8:
            return "score-excellent"
        elif score >= 0.6:
            return "score-good"
        elif score >= 0.4:
            return "score-fair"
        else:
            return "score-poor"
    
    def _cleanup_old_data(self):
        """Clean up old data from database."""
        # Keep data for last 30 days
        cutoff_time = datetime.now() - timedelta(days=30)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clean old metrics
            cursor.execute('''
                DELETE FROM quality_metrics
                WHERE timestamp < ?
            ''', (cutoff_time.isoformat(),))
            
            # Clean resolved alerts older than 7 days
            alert_cutoff = datetime.now() - timedelta(days=7)
            cursor.execute('''
                DELETE FROM quality_alerts
                WHERE timestamp < ? AND resolved = TRUE
            ''', (alert_cutoff.isoformat(),))
            
            conn.commit()
    
    def export_quality_report(self, days_back: int = 7) -> str:
        """Export comprehensive quality report."""
        dashboard_data = self.get_dashboard_data(hours_back=days_back * 24)
        
        report = []
        report.append("=" * 80)
        report.append("QUALITY MONITORING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Period: Last {days_back} days")
        report.append("")
        
        # Performance summary
        summary = dashboard_data.performance_summary
        report.append("PERFORMANCE SUMMARY:")
        report.append(f"  Total papers processed: {summary['total_papers']}")
        report.append(f"  Average overall score: {summary['avg_overall_score']:.2f}")
        report.append("  Grade distribution:")
        for grade, count in summary['score_distribution'].items():
            report.append(f"    {grade}: {count}")
        report.append("")
        
        # Current metrics
        report.append("CURRENT QUALITY METRICS:")
        for metric_name, metric in dashboard_data.current_metrics.items():
            report.append(f"  {metric_name}: {metric.value:.2f}")
        report.append("")
        
        # Active alerts
        if dashboard_data.active_alerts:
            report.append("ACTIVE ALERTS:")
            for alert in dashboard_data.active_alerts:
                report.append(f"  {alert.severity.upper()}: {alert.message}")
        report.append("")
        
        # Quality trends
        if dashboard_data.quality_trends:
            report.append("QUALITY TRENDS:")
            for trend in dashboard_data.quality_trends:
                report.append(f"  {trend.metric_name}: {trend.trend_direction} "
                             f"(confidence: {trend.confidence:.2f})")
        report.append("")
        
        # Recent papers
        if dashboard_data.recent_papers:
            report.append("RECENT PAPERS (Top 10):")
            for paper in dashboard_data.recent_papers[:10]:
                report.append(f"  {paper.paper_id[:20]}: {paper.overall_score:.2f} "
                             f"(Grade: {paper.quality_grade})")
        
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.dashboard_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def get_dashboard_url(self) -> str:
        """Get URL to access the dashboard."""
        dashboard_file = self.dashboard_dir / "quality_dashboard.html"
        return f"file://{dashboard_file.absolute()}"
