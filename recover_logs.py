import os
import json
import glob

def recover_logs():
    results_dir = 'results'
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    jsonl_files = glob.glob(os.path.join(results_dir, '*.jsonl'))
    
    for f_path in jsonl_files:
        basename = os.path.basename(f_path).replace('.jsonl', '.log')
        log_path = os.path.join(logs_dir, f'recovered_{basename}')
        
        with open(f_path, 'r') as f_in, open(log_path, 'w') as f_out:
            for line in f_in:
                if not line.strip(): continue
                data = json.loads(line)
                
                f_out.write("="*60 + "\n")
                f_out.write(f"SCENARIO {data.get('scenario_id', '?')} | Bias: {data.get('doctor_bias', 'None')}\n")
                f_out.write("="*60 + "\n\n")
                
                if 'full_dialogue' in data:
                    for turn in data['full_dialogue']:
                        role = turn.get('role', 'unknown').capitalize()
                        t_num = turn.get('turn', '?')
                        msg = turn.get('message', '')
                        
                        if 'differential' in turn:
                            f_out.write(f"\n  [METACOGNITION] Differential: {turn['differential']}\n")
                            
                        f_out.write(f"\n  {role} [Turn {t_num}]: {msg}\n")
                
                f_out.write("\n  Doctor's diagnosis complete.\n")
                f_out.write(f"  Doctor's Diagnosis:  {data.get('predicted_diagnosis', '')}\n")
                f_out.write(f"  Ground Truth:        {data.get('ground_truth', '')}\n")
                
                correct_str = '✅ CORRECT' if data.get('correct', False) else '❌ INCORRECT'
                f_out.write(f"  Result:              {correct_str}\n")
                
                if 'metrics' in data:
                    f_out.write(f"  Metrics:             {data['metrics']}\n")
                    
        print(f"Recovered log saved to: {log_path}")

if __name__ == '__main__':
    recover_logs()
