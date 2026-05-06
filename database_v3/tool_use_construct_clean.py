import json


input_file = 'dataset/tool_use_database_all_cases.jsonl'
output_file = 'dataset/cleaned_tool_use_database_all_cases.jsonl'

# Open the input file and read the data
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        record = json.loads(line.strip())
        
        # Extract the Python code from the completion field
        start = record['completion'].find('<python>')
        end = record['completion'].find('</python>') + len('</python>')
        
        # Keep only the Python code
        record['completion'] = record['completion'][start:end]
        
        # Write the cleaned record to the output file
        outfile.write(json.dumps(record) + '\n')

print(f'Cleaned data has been written to {output_file}.')
