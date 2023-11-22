use candle_core::{CudaDevice, DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, VarBuilder, VarMap};
use huggingface_candle_transformer::transformer::masking::{get_mask, masked_fill};
use huggingface_candle_transformer::transformer::mh_attention::MHAttention;
fn main() -> Result<()> {
    let device = Device::Cpu;
    let m = get_mask(10, &device)?;
    println!("m:{}", m);

    let scores = Tensor::rand(0.0, 1.0, (2, 8, 10, 10), &device)?.to_dtype(DType::F32)?;
    let mask = m.unsqueeze(0)?.unsqueeze(0)?.repeat((2, 8))?;
    println!("mask:{:?}", mask.shape());

    let mscores = masked_fill(&scores, &mask, f32::NEG_INFINITY)?;

    println!("mscores:{}", mscores.i(0)?.i(0)?);

    let smscores = candle_nn::ops::softmax(&mscores, D::Minus1)?;
    println!("smscores:{}", smscores.i(0)?.i(0)?);

    Ok(())
}
