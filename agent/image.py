import urllib.parse

def generate_image(prompt: str) -> str:
    """Generate an image URL using Pollinations AI."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    
    # URL encode the prompt text
    encoded_prompt = urllib.parse.quote(prompt.strip())
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    
    return image_url