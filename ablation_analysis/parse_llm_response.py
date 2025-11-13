def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response and extract decision, confidence, and reasoning."""
    result = {
        'decision': None,
        'confidence': None,
        'reasoning': 'Failed to parse response'
    }

    if not response:
        return result

    try:
        # Try to parse as JSON
        response_clean = response.strip()

        # Remove any markdown code blocks if present
        if response_clean.startswith('```'):
            response_clean = response_clean.split('```')[1]
            if response_clean.startswith('json'):
                response_clean = response_clean[4:]

        try:
            json_data = json.loads(response_clean)
            if isinstance(json_data, dict):
                result['decision'] = json_data.get('DECISION')
                result['confidence'] = json_data.get('CONFIDENCE')
                result['reasoning'] = json_data.get('REASONING', 'No reasoning provided')
                return result
        except json.JSONDecodeError:
            # Fall back to line-by-line parsing
            pass

        # Parse line by line for non-JSON responses
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('DECISION:'):
                decision = line.replace('DECISION:', '').strip()
                if decision in ['Yes', 'No']:
                    result['decision'] = decision
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = int(line.replace('CONFIDENCE:', '').strip())
                    if 1 <= confidence <= 10:
                        result['confidence'] = confidence
                except ValueError:
                    pass
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.replace('REASONING:', '').strip()

        return result

    except Exception as e:
        logging.error(f"Error parsing structured response: {e}")
        return result