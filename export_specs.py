import json
import yaml
import requests

# Get the OpenAPI spec from your running server
response = requests.get("http://127.0.0.1:8000/openapi.json")
openapi_spec = response.json()

# Save as YAML
with open("besesr_api_specs.yaml", "w") as f:
    yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)

print("✅ API specs exported to besesr_api_specs.yaml")

# Also save as JSON for backup
with open("besesr_api_specs.json", "w") as f:
    json.dump(openapi_spec, f, indent=2)

print("✅ API specs also saved as besesr_api_specs.json")