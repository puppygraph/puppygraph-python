# PuppyGraphAgent for Mining Site (Databricks Blog)

## Overview
This project demonstrates the use of PuppyGraphAgent for processing and analyzing data in a mining site environment using Databricks.

## Tables

- **Work Orders**: `pg_databricks.gold.work_orders`
- **Troubleshooting Guide**: `pg_databricks.silver.troubleshooting_guide`
- **Assets**: `pg_databricks.silver.assets`
- **Failure Type**: `pg_databricks.bronze.failure_type`

## How to Run

1. Set up the schema connection to PuppyGraph:
   ```bash
   python set_schema.py
   ```
2. Run the graph agent:
   ```bash
   python run_agent.py
   ```

## Requirements

See [tool.poetry.group.app.dependencies] for the required dependencies.
