# Makefile for MoE Data Reorganizer
# 
# This Makefile provides convenient commands for running the data reorganization
# process and related operations.

PYTHON := python3
BASE_DIR := ..
SCRIPT_DIR := auto_handle_data

# Default target
.PHONY: help
help:
	@echo "MoE Data Reorganizer - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Main Operations:"
	@echo "  make run          - Run reorganizer in interactive mode"
	@echo "  make auto         - Run reorganizer in automatic mode"
	@echo "  make reorganize   - Run reorganization directly"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make check        - Check source directories and configuration"
	@echo "  make clean        - Clean up log files and temporary data"
	@echo "  make status       - Show current status and file counts"
	@echo ""
	@echo "Examples:"
	@echo "  make run BASE_DIR=/path/to/project"
	@echo "  make auto PYTHON=python3.9"

# Main operations
.PHONY: run
run:
	@echo "🎯 Starting MoE Data Reorganizer (Interactive Mode)"
	cd $(BASE_DIR) && $(PYTHON) $(SCRIPT_DIR)/run_reorganizer.py

.PHONY: auto
auto:
	@echo "🤖 Starting MoE Data Reorganizer (Automatic Mode)"
	cd $(BASE_DIR) && $(PYTHON) $(SCRIPT_DIR)/run_reorganizer.py --auto

.PHONY: reorganize
reorganize:
	@echo "🚀 Running MoE Data Reorganizer"
	cd $(BASE_DIR) && $(PYTHON) $(SCRIPT_DIR)/moe_data_reorganizer.py

# Utility commands
.PHONY: check
check:
	@echo "🔍 Checking Source Directories"
	@echo ""
	@echo "Source Directory Status:"
	@ls -la $(BASE_DIR)/moe_mg/node1/realistic_trace/ 2>/dev/null | head -5 || echo "❌ Node1 realistic_trace not found"
	@ls -la $(BASE_DIR)/moe_mg/node2/realistic_trace/ 2>/dev/null | head -5 || echo "❌ Node2 realistic_trace not found"
	@ls -la $(BASE_DIR)/moe_mg/node1/profiler_log/ 2>/dev/null | head -5 || echo "❌ Node1 profiler_log not found"

.PHONY: clean
clean:
	@echo "🧹 Cleaning up log files and temporary data"
	rm -f $(BASE_DIR)/moe_data_reorganizer.log
	rm -f $(BASE_DIR)/moe_reorganizer_run.log
	rm -f $(BASE_DIR)/$(SCRIPT_DIR)/*.pyc
	rm -rf $(BASE_DIR)/$(SCRIPT_DIR)/__pycache__
	@echo "✅ Cleanup completed"

.PHONY: status
status:
	@echo "📊 Current Status Report"
	@echo "======================="
	@echo ""
	@echo "Source Data:"
	@find $(BASE_DIR)/moe_mg/node1/realistic_trace/ -name "*.txt" 2>/dev/null | wc -l | xargs echo "  Node1 realistic files:"
	@find $(BASE_DIR)/moe_mg/node2/realistic_trace/ -name "*.txt" 2>/dev/null | wc -l | xargs echo "  Node2 realistic files:"
	@find $(BASE_DIR)/moe_mg/node1/profiler_log/ -name "*.txt" 2>/dev/null | wc -l | xargs echo "  Profiler files:"
	@echo ""
	@echo "Target Data:"
	@find $(BASE_DIR)/simulation_inputs/megatron_operation_log/new_moe/ -name "*.txt" 2>/dev/null | wc -l | xargs echo "  Reorganized files:" || echo "  Reorganized files: 0"
	@ls -1 $(BASE_DIR)/simulation_inputs/megatron_operation_log/new_moe/ 2>/dev/null | grep -c "setting" | xargs echo "  Settings created:" || echo "  Settings created: 0"

# Quick shortcuts
.PHONY: r
r: run

.PHONY: s
s: status

.PHONY: c
c: check
