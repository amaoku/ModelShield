# Watermark Generation Script

generating watermarked responses using OpenAI's language model (you can also use it to other LMaaS). The generated watermarked text includes special watermark words embedded in the answers to provided questions.


## Input and Output

### Input
- **Questions File**: A file containing a list of questions in JSON format, with each question having an `id` and `instruction`.
- **API Key**: OpenAI API key(s) for generating responses.

### Output
- A JSON file storing results for each processed question:
  - Question ID and content.
  - Generated text with embedded watermark words.
  - Original question and model prediction.

## Usage

python watermark_generation.py
