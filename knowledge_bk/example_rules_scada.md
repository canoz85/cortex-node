Title: SCADA Alarm Threshold Policy
Policy ID: SCADA-ALARM-TH-2026-06
Status: Updated
Effective Date: 2026-06-28
Owner: OT Operations

Scope
This policy defines warning and critical alarm thresholds for live process signals in SCADA production lines A and B.

Thresholds

Reactor Temperature
Warning: 78.0 C
Critical: 82.0 C
Clear threshold: below 76.5 C for 30 seconds
Line Pressure
Warning: 6.2 bar
Critical: 6.8 bar
Clear threshold: below 6.0 bar for 20 seconds
Tank Level
Warning Low: 22 percent
Critical Low: 15 percent
Warning High: 88 percent
Critical High: 93 percent
Clear threshold: inside 25 to 85 percent for 30 seconds
Alarm behavior rules

Apply hysteresis before clearing alarms to prevent alarm chatter.
Require 2 consecutive bad samples at 1 second interval before raising critical.
Suppress duplicate alarm notifications for 60 seconds.
Escalate unresolved critical alarms to on-call engineer after 120 seconds.
Operator response requirements

Warning alarms: acknowledge within 5 minutes.
Critical alarms: acknowledge within 1 minute and add operator note.
Every threshold violation must record timestamp, tag, previous value, new value, and operator action.
Change log

2026-06-28: Pressure critical changed from 7.0 to 6.8 bar due to pump seal incidents.
2026-06-28: Reactor warning changed from 80.0 to 78.0 C.
2026-06-28: Added duplicate suppression window of 60 seconds.
Example JSON knowledge entry

{
"policy_id": "SCADA-ALARM-TH-2026-06",
"title": "SCADA Alarm Threshold Policy",
"status": "updated",
"effective_date": "2026-06-28",
"equipment_scope": ["line_a", "line_b"],
"thresholds": {
"reactor_temperature_c": {
"warning": 78.0,
"critical": 82.0,
"clear_below": 76.5,
"clear_seconds": 30
},
"line_pressure_bar": {
"warning": 6.2,
"critical": 6.8,
"clear_below": 6.0,
"clear_seconds": 20
},
"tank_level_percent": {
"warning_low": 22,
"critical_low": 15,
"warning_high": 88,
"critical_high": 93,
"clear_range": [25, 85],
"clear_seconds": 30
}
},
"alarm_rules": {
"critical_consecutive_samples": 2,
"sample_interval_seconds": 1,
"duplicate_suppression_seconds": 60,
"critical_escalation_seconds": 120
},
"changes": [
{
"date": "2026-06-28",
"field": "line_pressure_bar.critical",
"old": 7.0,
"new": 6.8,
"reason": "pump seal incidents"
}
]
}