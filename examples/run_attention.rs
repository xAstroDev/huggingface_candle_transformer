use candle_core::{CudaDevice, DType, Device, Module, Result, Tensor};
use candle_nn::{linear, VarBuilder, VarMap};
use huggingface_candle_transformer::transformer::{
    masking::create_seq_mask, mh_attention::MHAttention,
};
fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let d_model: usize = 128;
    let num_heads: usize = 8;
    let model = MHAttention::new(&vb, d_model, num_heads, true, true, true)?;

    let q = Tensor::rand(0.0, 1.0, &[2, 4, d_model], &Device::Cpu)?.to_dtype(DType::F32)?;
    let kv = Tensor::rand(0.0, 1.0, &[2, 10, d_model], &Device::Cpu)?.to_dtype(DType::F32)?;

    let mask = create_seq_mask(&[2, 4, 6, 8], 10, &Device::Cpu)?;

    let out = model.forward(&q, &kv, Some(&mask))?;

    println!("out:{:?}", out);

    Ok(())
}
