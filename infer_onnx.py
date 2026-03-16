import os
import onnxruntime as ort
import numpy as np
import soundfile as sf

def infer_onnx(onnx_path: str, input_npy_path: str, output_wav_path: str, sample_rate: int = 24000):
    print(f"Loading ONNX model from: {onnx_path}")
    

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    print(f"Model expects input name: '{input_name}', shape: {session.get_inputs()[0].shape}")
    
    print(f"Loading input data from: {input_npy_path}")
    codes = np.load(input_npy_path)
    
    if codes.ndim == 2:
        codes = np.expand_dims(codes.transpose(1, 0), axis=0)
    elif codes.ndim == 3 and codes.shape[0] != 1 and codes.shape[2] != 16:
        pass
        
    codes = codes.astype(np.int64)
    print(f"Processed input shape for ONNX: {codes.shape}")

    print("Running inference...")
    outputs = session.run(["wav"], {input_name: codes})
    
    wav_tensor = outputs[0]
    print(f"Output wav shape: {wav_tensor.shape}")
    
    wav_1d = wav_tensor.flatten()
    
    print(f"Saving audio to {output_wav_path} (Sample Rate: {sample_rate} Hz)")
    sf.write(output_wav_path, wav_1d, sample_rate, format='WAV', subtype='PCM_16')
    
    print("Inference finished successfully!")

if __name__ == "__main__":

    ONNX_MODEL_PATH = "qwen3_tts_12hz_decoder_sim.onnx"
    NPY_FILE_PATH = "debug_output/sample_0_codes.npy" 
    OUTPUT_WAV_FILE = "debug_output/onnx_output.wav"
    
    infer_onnx(ONNX_MODEL_PATH, NPY_FILE_PATH, OUTPUT_WAV_FILE, sample_rate=24000)