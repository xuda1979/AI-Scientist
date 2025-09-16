"""
Quality Enhancements Package
============================

Advanced quality assessment and improvement modules for research paper generation.

This package provides comprehensive quality enhancement capabilities including:
- Enhanced quality metrics with semantic analysis
- Multi-stage peer review simulation
- Advanced literature integration and gap analysis
- Visual content validation for figures and tables
- Simulation methodology validation
- Anti-hallucination fact checking
- Intelligent content enhancement
- Real-time quality monitoring dashboard

All modules are designed to work with the main sciresearch workflow to
significantly improve research paper quality through automated analysis
and enhancement suggestions.
"""

try:
    # Core quality analysis
    from .enhanced_metrics import EnhancedQualityAnalyzer, QualityMetrics
    HAS_ENHANCED_METRICS = True
except ImportError as e:
    print(f"⚠ Failed to import enhanced_metrics: {e}")
    HAS_ENHANCED_METRICS = False

try:
    # Multi-stage review system  
    from .multi_stage_review import MultiStageReviewer, ReviewResult
    HAS_MULTI_STAGE_REVIEW = True
except ImportError as e:
    print(f"⚠ Failed to import multi_stage_review: {e}")
    HAS_MULTI_STAGE_REVIEW = False

try:
    # Literature integration
    from .literature_integration import LiteratureIntegrator, LiteratureAnalysisResult
    HAS_LITERATURE_INTEGRATION = True
except ImportError as e:
    print(f"⚠ Failed to import literature_integration: {e}")
    HAS_LITERATURE_INTEGRATION = False

try:
    # Visual content validation
    from .visual_content_validator import VisualContentValidator, VisualValidationResult
    HAS_VISUAL_VALIDATOR = True
except ImportError as e:
    print(f"⚠ Failed to import visual_content_validator: {e}")
    HAS_VISUAL_VALIDATOR = False

try:
    # Advanced simulation validation
    from .advanced_simulation_validator import AdvancedSimulationValidator, SimulationValidationResult
    HAS_SIMULATION_VALIDATOR = True
except ImportError as e:
    print(f"⚠ Failed to import advanced_simulation_validator: {e}")
    HAS_SIMULATION_VALIDATOR = False

try:
    # Anti-hallucination validation
    from .anti_hallucination_validator import AntiHallucinationValidator, HallucinationValidationResult
    HAS_ANTI_HALLUCINATION = True
except ImportError as e:
    print(f"⚠ Failed to import anti_hallucination_validator: {e}")
    HAS_ANTI_HALLUCINATION = False

try:
    # Intelligent content enhancement
    from .intelligent_content_enhancer import IntelligentContentEnhancer, ContentEnhancementResult
    HAS_CONTENT_ENHANCER = True
except ImportError as e:
    print(f"⚠ Failed to import intelligent_content_enhancer: {e}")
    HAS_CONTENT_ENHANCER = False

try:
    # Quality monitoring dashboard
    from .quality_monitor_dashboard import QualityMonitorDashboard, QualityDashboardData
    HAS_QUALITY_MONITOR = True
except ImportError as e:
    print(f"⚠ Failed to import quality_monitor_dashboard: {e}")
    HAS_QUALITY_MONITOR = False


# Build __all__ list based on successful imports
__all__ = []

if HAS_ENHANCED_METRICS:
    __all__.extend(['EnhancedQualityAnalyzer', 'QualityMetrics'])

if HAS_MULTI_STAGE_REVIEW:
    __all__.extend(['MultiStageReviewer', 'ReviewResult'])

if HAS_LITERATURE_INTEGRATION:
    __all__.extend(['LiteratureIntegrator', 'LiteratureAnalysisResult'])

if HAS_VISUAL_VALIDATOR:
    __all__.extend(['VisualContentValidator', 'VisualValidationResult'])

if HAS_SIMULATION_VALIDATOR:
    __all__.extend(['AdvancedSimulationValidator', 'SimulationValidationResult'])

if HAS_ANTI_HALLUCINATION:
    __all__.extend(['AntiHallucinationValidator', 'HallucinationValidationResult'])

if HAS_CONTENT_ENHANCER:
    __all__.extend(['IntelligentContentEnhancer', 'ContentEnhancementResult'])

if HAS_QUALITY_MONITOR:
    __all__.extend(['QualityMonitorDashboard', 'QualityDashboardData'])


def get_available_modules():
    """Get list of successfully imported quality enhancement modules."""
    available = []
    
    if HAS_ENHANCED_METRICS:
        available.append('enhanced_metrics')
    if HAS_MULTI_STAGE_REVIEW:
        available.append('multi_stage_review')
    if HAS_LITERATURE_INTEGRATION:
        available.append('literature_integration')
    if HAS_VISUAL_VALIDATOR:
        available.append('visual_content_validator')
    if HAS_SIMULATION_VALIDATOR:
        available.append('advanced_simulation_validator')
    if HAS_ANTI_HALLUCINATION:
        available.append('anti_hallucination_validator')
    if HAS_CONTENT_ENHANCER:
        available.append('intelligent_content_enhancer')
    if HAS_QUALITY_MONITOR:
        available.append('quality_monitor_dashboard')
    
    return available


def get_quality_enhancement_summary():
    """Get summary of available quality enhancement capabilities."""
    available_modules = get_available_modules()
    
    summary = {
        'total_modules': len(available_modules),
        'available_modules': available_modules,
        'capabilities': {
            'semantic_analysis': HAS_ENHANCED_METRICS,
            'peer_review_simulation': HAS_MULTI_STAGE_REVIEW,
            'literature_analysis': HAS_LITERATURE_INTEGRATION,
            'visual_validation': HAS_VISUAL_VALIDATOR,
            'simulation_validation': HAS_SIMULATION_VALIDATOR,
            'fact_checking': HAS_ANTI_HALLUCINATION,
            'content_enhancement': HAS_CONTENT_ENHANCER,
            'quality_monitoring': HAS_QUALITY_MONITOR
        }
    }
    
    return summary


# Version and metadata
__version__ = "2.0.0"
__author__ = "SciResearch Workflow Enhancement Team"
