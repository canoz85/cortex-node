"""SAP operations tools for querying ABAP tables and executing reports."""

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def query_abap_table(
    table_name: str,
    fields: str = "*",
    where_clause: str = "",
    max_rows: int = 100,
) -> dict[str, Any]:
    """
    Query an ABAP table from SAP.

    Args:
        table_name: SAP table name (e.g., 'MARA' for materials, 'MARC' for material plant data)
        fields: Comma-separated field names, or '*' for all fields
        where_clause: WHERE condition (e.g., 'MATNR = "123456"')
        max_rows: Maximum number of rows to return (default 100)

    Returns:
        dict with 'data' (list of records), 'row_count', 'fields', and 'table_name'

    Examples:
        - query_abap_table(table_name='MARA', where_clause='MATNR = "123456"')
        - query_abap_table(table_name='MARC', fields='MATNR,WERKS,LGORT', max_rows=50)
    """
    # Validation
    if not table_name:
        return {"error": "table_name is required", "success": False}

    if not isinstance(max_rows, int) or max_rows < 1:
        return {"error": "max_rows must be a positive integer", "success": False}

    try:
        # Placeholder implementation - in production, this would connect to SAP
        # via RFC or OData API (e.g., pyrfc, requests to SAP Gateway)
        logger.info(f"Query ABAP table: {table_name} with where_clause: {where_clause}")

        # Mock response for demonstration
        if table_name.upper() == "MARA":
            mock_data = [
                {"MATNR": "123456", "MAKTX": "Example Material 1", "MEINS": "EA", "MTART": "FERT"},
                {"MATNR": "654321", "MAKTX": "Example Material 2", "MEINS": "PC", "MTART": "FERT"},
            ]
        elif table_name.upper() == "MARC":
            mock_data = [
                {"MATNR": "123456", "WERKS": "1000", "LGORT": "0001", "LABST": 100},
                {"MATNR": "123456", "WERKS": "2000", "LGORT": "0002", "LABST": 50},
            ]
        else:
            mock_data = []

        return {
            "success": True,
            "table_name": table_name,
            "row_count": len(mock_data),
            "fields": fields.split(",") if fields != "*" else ["*"],
            "data": mock_data[:max_rows],
        }

    except Exception as e:
        logger.error(f"Error querying table {table_name}: {e}")
        return {"error": str(e), "success": False, "table_name": table_name}


@tool
def execute_abap_report(
    report_name: str,
    parameters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Execute an ABAP report in SAP and retrieve results.

    Args:
        report_name: Name of the ABAP report (e.g., 'RMMG100', 'MFBF')
        parameters: Dictionary of report parameters (e.g., {'P_MATNR': '123456', 'P_WERKS': '1000'})

    Returns:
        dict with 'report_name', 'parameters_used', 'success', and 'output' or 'error'

    Examples:
        - execute_abap_report(report_name='RMMG100', parameters={'P_MATNR': '123456'})
        - execute_abap_report(report_name='MFBF', parameters={'P_WERKS': '1000', 'P_LGORT': '0001'})
    """
    if not report_name:
        return {"error": "report_name is required", "success": False}

    parameters = parameters or {}

    try:
        logger.info(f"Execute ABAP report: {report_name} with parameters: {parameters}")

        # Placeholder implementation - in production, this would connect to SAP
        # via RFC or OData API and execute the report
        mock_output = f"Report {report_name} executed with parameters {parameters}. Mock data returned."

        return {
            "success": True,
            "report_name": report_name,
            "parameters_used": parameters,
            "output": mock_output,
            "row_count": 0,
        }

    except Exception as e:
        logger.error(f"Error executing report {report_name}: {e}")
        return {"error": str(e), "success": False, "report_name": report_name}


@tool
def lookup_material(
    material_id: str,
    include_plant_data: bool = False,
) -> dict[str, Any]:
    """
    Look up material master data from SAP.

    Args:
        material_id: Material number (e.g., '123456')
        include_plant_data: If True, also retrieve plant-specific data from MARC table

    Returns:
        dict with material master info (MARA fields) and optionally plant data (MARC fields)

    Examples:
        - lookup_material(material_id='123456')
        - lookup_material(material_id='123456', include_plant_data=True)
    """
    if not material_id:
        return {"error": "material_id is required", "success": False}

    try:
        logger.info(f"Lookup material: {material_id}")

        # Query MARA (material master)
        mara_result = query_abap_table(
            table_name="MARA",
            fields="MATNR,MAKTX,MEINS,MTART,ERSDA,ERNAM,LAEDA,LANAME",
            where_clause=f'MATNR = "{material_id}"',
            max_rows=1,
        )

        if not mara_result.get("success") or not mara_result.get("data"):
            return {"error": f"Material {material_id} not found", "success": False}

        result = {
            "success": True,
            "material_id": material_id,
            "material_data": mara_result["data"][0] if mara_result["data"] else None,
        }

        # Optional: Query MARC (plant-specific data)
        if include_plant_data:
            marc_result = query_abap_table(
                table_name="MARC",
                fields="MATNR,WERKS,LGORT,LABST,UMLMC,UMLTB,UMLMT",
                where_clause=f'MATNR = "{material_id}"',
                max_rows=10,
            )
            result["plant_data"] = marc_result.get("data", [])

        return result

    except Exception as e:
        logger.error(f"Error looking up material {material_id}: {e}")
        return {"error": str(e), "success": False, "material_id": material_id}


@tool
def get_report_data(
    report_name: str,
    output_format: str = "json",
    parameters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Execute a report and export data in specified format.

    Args:
        report_name: Name of the ABAP report
        output_format: Format for export ('json', 'csv', 'xlsx')
        parameters: Report parameters as dictionary

    Returns:
        dict with 'format', 'data', 'row_count', 'success', and optional 'file_path'

    Examples:
        - get_report_data(report_name='RMMG100', output_format='json', parameters={'P_MATNR': '123456'})
        - get_report_data(report_name='MFBF', output_format='csv', parameters={'P_WERKS': '1000'})
    """
    if not report_name:
        return {"error": "report_name is required", "success": False}

    if output_format not in ("json", "csv", "xlsx"):
        return {"error": f"output_format must be 'json', 'csv', or 'xlsx', got {output_format}", "success": False}

    parameters = parameters or {}

    try:
        logger.info(f"Get report data: {report_name} in {output_format} format")

        # Execute the report
        report_result = execute_abap_report(report_name, parameters)

        if not report_result.get("success"):
            return report_result

        # Format the output
        mock_data = [
            {"field1": "value1", "field2": "value2"},
            {"field1": "value3", "field2": "value4"},
        ]

        return {
            "success": True,
            "report_name": report_name,
            "output_format": output_format,
            "row_count": len(mock_data),
            "data": mock_data,
        }

    except Exception as e:
        logger.error(f"Error exporting report {report_name}: {e}")
        return {"error": str(e), "success": False, "report_name": report_name}


def get_sap_tools(workspace_dir: str = "workspace") -> list:
    """
    Return list of SAP operation tools available to the agent.

    Args:
        workspace_dir: Workspace directory (for future enhancements)

    Returns:
        List of tool objects
    """
    return [
        query_abap_table,
        execute_abap_report,
        lookup_material,
        get_report_data,
    ]
