# Changelog

All notable changes to the Enhanced SciResearch Workflow project will be documented in this file.

## [Unreleased]

### Removed
- Eliminated ag-qec specific scripts and Monte Carlo artifact files so the workflow remains focused on generating general-purpose papers.

### Changed
- Updated the Windows helper batch script to accept arbitrary topics, fields, and research questions instead of targeting a single project.

## [2.0.0] - 2025-09-03

### Added
- **Multi-Model AI Support**: Added support for Google AI (Gemini 1.5 Pro) alongside OpenAI models
- **Universal Chat Interface**: Automatic provider detection and routing for different AI models
- **Proxy Configuration**: Automatic proxy setup for Gemini models behind firewalls
- **Enhanced OpenAI Features**: Added reasoning effort and verbosity controls, web search tools
- **Production-Ready Infrastructure**: Comprehensive logging, error handling, and resource management
- **Configuration Management**: JSON-based configuration with environment variable support
- **Quality Validation System**: Automated quality scoring and iterative improvement
- **Security Validation**: Code security analysis before execution
- **Reference Validation**: External reference checking with DOI validation
- **Advanced Argument Handling**: Fixed topic prompts for existing paper modification

### Enhanced
- **LaTeX Compilation**: Improved compilation with automatic error detection and fixing
- **Simulation Integration**: Better code extraction and execution with result integration
- **Error Classification**: Intelligent error handling with automatic retries
- **Fallback Mechanisms**: Multi-model fallback chains for reliability
- **User Experience**: Streamlined command-line interface and interactive prompts

### Fixed
- **Argument Validation**: No more topic prompts when using --modify-existing
- **Model Routing**: Proper API routing for different model providers
- **Proxy Handling**: Correct proxy configuration only for models that need it
- **Memory Management**: Better resource cleanup and memory limits

### Infrastructure
- **Project Cleanup**: Organized documentation, removed temporary files
- **Enhanced README**: Comprehensive documentation with examples
- **Dependency Management**: Added requirements.txt and proper .gitignore
- **Code Organization**: Moved documentation to docs/ directory

## [1.0.0] - Initial Release

### Added
- Basic workflow for paper generation and review
- OpenAI GPT integration
- LaTeX compilation and validation
- Simple command-line interface
