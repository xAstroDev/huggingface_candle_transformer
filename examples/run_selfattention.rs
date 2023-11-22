use candle_core::{CudaDevice, DType, Device, Module, Result, Tensor};
use candle_nn::{linear, VarBuilder, VarMap};
use huggingface_candle_transformer::transformer::{
    masking::get_mask, mh_selfattention::MHSelfAttention,
};
fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let dim: usize = 64;
    let num_heads: usize = 8;
    let model = MHSelfAttention::new(&vb, dim, num_heads, true, true)?;

    let img = Tensor::rand(0.0, 1.0, &[2, 4, 64], &Device::Cpu)?.to_dtype(DType::F32)?;

    let mask = get_mask(4, &Device::Cpu)?;

    let out = model.forward(&img, Some(&mask))?;

    println!("out shape:{:?}", out.shape());
    println!("out:{:?}", out);

    Ok(())
}
