import onnx
from onnx import helper
import numpy as np

def remove_isnan_where(onnx_path, out_onnx_path):
    print("Loading graph...")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    output_to_producer = {}
    for node in graph.node:
        for output in node.output:
            output_to_producer[output] = node
    
    nodes_to_remove = []
    replacements = {}
    
    print("Scanning for IsNaN->Where patterns...")
    
    for node in graph.node:
        if node.op_type != "Where":
            continue
        
        if len(node.input) < 3:
            continue
        
        condition_name = node.input[0]
        
        if condition_name not in output_to_producer:
            continue
        
        producer = output_to_producer[condition_name]
        
        if producer.op_type == "IsNaN":
            print(f"  Found: IsNaN({producer.name}) -> Where({node.name})")
            

            input_1_name = node.input[1]
            input_2_name = node.input[2]
            
            real_tensor_name = input_2_name
            
            for initializer in graph.initializer:
                if initializer.name == input_1_name:
                    real_tensor_name = input_2_name
                    break
                elif initializer.name == input_2_name:
                    real_tensor_name = input_1_name
                    break
            
            where_output = node.output[0]
            replacements[where_output] = real_tensor_name
            
            nodes_to_remove.append(node.name)
            nodes_to_remove.append(producer.name)
    
    if not nodes_to_remove:
        print("No IsNaN->Where patterns found.")
        onnx.save(model, out_onnx_path)
        print(f"Saved model to {out_onnx_path}")
        return
    
    print(f"Removing {len(nodes_to_remove)} nodes...")
    
    new_nodes = []
    for node in graph.node:
        if node.name in nodes_to_remove:
            continue
        
        new_inputs = []
        for inp in node.input:
            new_inputs.append(replacements.get(inp, inp))
        
        new_node = helper.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=list(node.output),
            name=node.name,
            domain=node.domain,
        )
        
        for attr in node.attribute:
            new_node.attribute.append(attr)
        
        new_nodes.append(new_node)
    
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    onnx.checker.check_model(model)
    
    onnx.save(model, out_onnx_path)
    print(f"Successfully removed {len(nodes_to_remove)} nodes.")
    print(f"Saved optimized model to {out_onnx_path}")

if __name__ == "__main__":
    remove_isnan_where(
        "qwen3_tts_12hz_decoder_static_sim.onnx",
        "qwen3_tts_12hz_decoder_clean.onnx"
    )
