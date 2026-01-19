from lambda_function import lambda_handler

# 1. Simulate the "Event" that AWS would send
# (Replace with an actual image path from your test folder)
test_event = {
    'url': 'data/test/PNEUMONIA/person100_bacteria_475.jpeg'
}

# 2. Trigger the handler
print("Testing Lambda Handler...")
response = lambda_handler(test_event, None)

# 3. See the output
print(f"Prediction: {response['prediction']}")
print(f"Probability: {response['probability']:.4f}")