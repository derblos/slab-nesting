"""
Test script for the Nesting Tool API

Run this after starting the API server to verify it's working correctly.

Usage:
    1. Start API: uvicorn backend.main:app --reload
    2. Run tests: python test_api.py
"""

import requests
import json

API_URL = "http://localhost:8000"


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200, "Health check failed!"
    print("✅ Health check passed!")


def test_create_parts():
    """Test creating parts"""
    print_section("Creating Test Parts")

    parts = [
        {
            "id": "part-1",
            "label": "Countertop",
            "qty": 1,
            "shape_type": "rect",
            "width": 134.75,
            "height": 53.5,
            "allow_rotation": True,
            "meta": {}
        },
        {
            "id": "part-2",
            "label": "Backsplash",
            "qty": 2,
            "shape_type": "rect",
            "width": 57.88,
            "height": 22.38,
            "allow_rotation": True,
            "meta": {}
        },
        {
            "id": "part-3",
            "label": "Island Top",
            "qty": 1,
            "shape_type": "rect",
            "width": 52.75,
            "height": 24.38,
            "allow_rotation": True,
            "meta": {}
        }
    ]

    for part in parts:
        response = requests.post(f"{API_URL}/api/parts", json=part)
        print(f"Created: {part['label']} - Status: {response.status_code}")
        assert response.status_code == 201, f"Failed to create {part['label']}"

    print("✅ All parts created successfully!")


def test_get_parts():
    """Test getting all parts"""
    print_section("Retrieving All Parts")

    response = requests.get(f"{API_URL}/api/parts")
    parts = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Total Parts: {len(parts)}")
    for part in parts:
        print(f"  - {part['label']}: {part['width']}\" × {part['height']}\" (Qty: {part['qty']})")

    assert response.status_code == 200, "Failed to get parts"
    assert len(parts) >= 3, "Expected at least 3 parts"
    print("✅ Parts retrieved successfully!")


def test_nesting():
    """Test nesting algorithm"""
    print_section("Running Nesting Algorithm")

    # Get current parts
    parts_response = requests.get(f"{API_URL}/api/parts")
    parts = parts_response.json()

    # Configure nesting
    config = {
        "sheet_w": 139.0,
        "sheet_h": 80.0,
        "clearance": 0.25,
        "allow_rotation": True,
        "autosplit_rects": False,
        "seam_gap": 0.125,
        "min_leg": 6.0,
        "prefer_long_split": True,
        "enable_L_seams": True,
        "L_max_leg": 48.0,
        "grid_step": 0.5,
        "units": "in",
        "precision": 2
    }

    # Run nesting
    nest_request = {
        "parts": parts,
        "config": config,
        "mode": "rectpack"
    }

    response = requests.post(f"{API_URL}/api/nest", json=nest_request)
    result = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"\nNesting Results:")
    print(f"  Sheets Used: {result['num_sheets']}")
    print(f"  Utilization: {result['utilization'] * 100:.2f}%")
    print(f"  Area Used: {result['total_area_used']:.2f} sq in")
    print(f"  Total Area: {result['total_area_available']:.2f} sq in")

    for i, sheet in enumerate(result['sheets'], 1):
        print(f"\n  Sheet {i}: {len(sheet['placements'])} parts")
        for p in sheet['placements']:
            print(f"    - {p['label']}: {p['w']:.2f}\" × {p['h']:.2f}\"")

    assert response.status_code == 200, "Nesting failed"
    assert result['num_sheets'] > 0, "No sheets generated"
    print("\n✅ Nesting completed successfully!")


def test_cleanup():
    """Clean up test data"""
    print_section("Cleaning Up Test Data")

    response = requests.delete(f"{API_URL}/api/parts")
    result = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Message: {result['message']}")
    print("✅ Cleanup completed!")


def run_all_tests():
    """Run all API tests"""
    try:
        print(f"\n🚀 Starting API Tests")
        print(f"API URL: {API_URL}")

        test_health()
        test_create_parts()
        test_get_parts()
        test_nesting()
        test_cleanup()

        print(f"\n{'='*60}")
        print(f"  🎉 ALL TESTS PASSED!")
        print(f"{'='*60}\n")

    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Could not connect to API at {API_URL}")
        print("Make sure the API server is running:")
        print("  uvicorn backend.main:app --reload")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
