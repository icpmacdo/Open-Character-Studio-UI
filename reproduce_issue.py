from character.constitution.schema import Meta

try:
    m = Meta(
        name="customer_service",
        description="A test description for verification."
    )
    print("SUCCESS: Meta instantiated successfully with 'customer_service'")
except Exception as e:
    print(f"FAILURE: {e}")
