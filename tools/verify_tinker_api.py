
import tinker

def test_client_creation():
    print("Initializing ServiceClient...")
    try:
        service_client = tinker.ServiceClient()
        print("ServiceClient initialized.")
    except Exception as e:
        print(f"Failed to init ServiceClient: {e}")
        return

    base_model = "Qwen/Qwen3-8B-Base" # Using a known small model from docs
    
    print(f"\nAttempting create_sampling_client(base_model='{base_model}')...")
    try:
        # This is the usage in the code I want to verify
        _client = service_client.create_sampling_client(base_model=base_model)
        print("Success! create_sampling_client accepts base_model.")
    except TypeError as e:
        print(f"TypeError: {e}")
        print("Checking if it accepts 'model_path' only...")
    except Exception as e:
        print(f"Failed with other error: {e}")

if __name__ == "__main__":
    test_client_creation()
