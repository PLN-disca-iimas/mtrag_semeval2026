"""
Script para convertir archivos de queries del formato de evaluación
a un formato compatible con reranking.py
"""

import json
import argparse
import os


def convert_clapnq_rewrite_to_queries(input_file: str, output_file: str):
    """
    Convierte clapnq_rewrite.jsonl al formato de queries para reranking.py
    
    Formato de entrada (clapnq_rewrite.jsonl):
    {"_id": "task_id", "text": "query text"}
    
    Formato de salida:
    {"task_id": "task_id", "query": "query text"}
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                
                # Extraer task_id y query
                task_id = item.get('_id', f'task_{line_num}')
                query = item.get('text', '')
                
                # Limpiar task_id si tiene formato "id<::>turn"
                if '<::>' in task_id:
                    task_id = task_id.replace('<::>', '::')
                
                if not query:
                    print(f"Warning: No query text found in line {line_num}, skipping...")
                    continue
                
                # Crear formato de salida
                output = {
                    "task_id": task_id,
                    "query": query
                }
                
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"✓ Converted queries written to {output_file}")


def convert_responses_to_queries(input_file: str, output_file: str):
    """
    Convierte archivos en formato responses-10.jsonl a queries
    
    Extrae el task_id y la query del campo 'input'
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                
                task_id = item.get('task_id', f'task_{line_num}')
                
                # Extraer query del campo input
                query = None
                if 'input' in item and isinstance(item['input'], list):
                    for inp in item['input']:
                        if isinstance(inp, dict) and inp.get('speaker') == 'user':
                            query = inp.get('text', '')
                            break
                
                if not query:
                    print(f"Warning: Could not extract query from line {line_num}, skipping...")
                    continue
                
                output = {
                    "task_id": task_id,
                    "query": query
                }
                
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"✓ Converted queries written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert evaluation query files to reranking.py format"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file (clapnq_rewrite.jsonl or responses format)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL file with queries for reranking.py"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["clapnq_rewrite", "responses"],
        default="clapnq_rewrite",
        help="Input file format (default: clapnq_rewrite)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if args.format == "clapnq_rewrite":
        convert_clapnq_rewrite_to_queries(args.input_file, args.output_file)
    elif args.format == "responses":
        convert_responses_to_queries(args.input_file, args.output_file)


if __name__ == "__main__":
    main()

