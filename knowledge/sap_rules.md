# SAP Development Standards and Best Practices

## Query Guidelines

### Table Selection
- Use **MARA** for material master general data (descriptions, units)
- Use **MARC** for plant-specific material data (storage locations, stock)
- Use **MAKT** for localized descriptions and names
- Use **EKPO** for purchase order items
- Use **LFA1** for vendor master data
- Use **KNA1** for customer master data

### Query Parameters
- Always include max_rows limit (default 100, max 1000) to prevent performance issues
- Use specific WHERE clauses to filter results early at SAP level
- Comma-separate field names for partial data retrieval instead of SELECT *
- Common WHERE conditions:
  - `MATNR = "XXXX"` - exact material lookup
  - `MAKTX LIKE '%pattern%'` - description search
  - `WERKS = "XXXX"` - filter by plant
  - `LIFNR = "XXXX"` - filter by vendor

### Common Field Abbreviations
- **MATNR** = Material Number
- **MAKTX** = Material Description (text)
- **MEINS** = Unit of Measure
- **MTART** = Material Type
- **WERKS** = Plant
- **LGORT** = Storage Location
- **LABST** = Unrestricted Stock
- **NETPR** = Net Price
- **MENGE** = Quantity
- **EBELN** = Purchase Order Number
- **VBELN** = Sales Order Number
- **KUNNR** = Customer Number
- **LIFNR** = Vendor/Supplier Number

## Report Execution

### Standard Reports
- **RMMG100** - Material Master Display and maintenance
  - Parameters: P_MATNR (material), P_WERKS (plant, optional)
- **MFBF** - Material Forecast Backflush
  - Parameters: P_WERKS (plant), P_MATNR (material)
- **RM06BOM** - Display Bill of Material
  - Parameters: P_MATNR (material), P_WERKS (plant), P_STLAN (BOM usage)

### Report Parameters
- Always provide required parameters; optional parameters can be omitted
- Use format: `{"PARAM_NAME": "value"}` for parameter dictionaries
- Values should match SAP data types (no leading zeros for numeric fields unless stored as text)

## Data Export

### Export Formats
- **JSON**: Best for API integration and structured data processing
- **CSV**: Best for Excel import and human review
- **XLSX**: Best for formatted reports with multiple sheets

### Export Best Practices
- Specify output_format parameter to control export type
- JSON is default for programmatic processing
- CSV for data migration to other systems
- Include relevant fields only to reduce export size

## Error Handling

### Common Errors
- **"Material not found"** - Verify material number format and case sensitivity
- **"Invalid plant"** - Check plant code in WHERE clause (e.g., plant "1000" vs "1")
- **"Table not found"** - Verify table name is correct (MARA, MARC, etc.)
- **"Field not recognized"** - Check field name spelling and table compatibility

### Recovery Actions
1. Check the error message for typos in material/table/plant codes
2. Verify field names are valid for the selected table
3. Use a simpler query with fewer WHERE conditions
4. Consult common_sap_tables reference in sap_examples.json

## Workflow Patterns

### Material Lookup Workflow
1. Call `lookup_material(material_id)` to get basic master data
2. If plant-specific data needed, call with `include_plant_data=True`
3. Parse response to extract relevant fields

### Material Search Workflow
1. Use `query_abap_table(table_name='MARA', where_clause=...)` for flexible searching
2. Provide specific search criteria (MATNR, MAKTX pattern, etc.)
3. Limit results with max_rows parameter

### Report Export Workflow
1. Identify required report (RMMG100, MFBF, etc.)
2. Prepare parameters dictionary
3. Call `execute_abap_report()` to run report
4. Call `get_report_data()` with desired output_format to export

## Integration Notes

- All queries return structured JSON with success/failure indicators
- Check `success: true/false` before processing response data
- Error messages are in `error` field on failure
- Row count is provided in `row_count` field for result sizing
- For failed queries, response includes the original request parameters for debugging
