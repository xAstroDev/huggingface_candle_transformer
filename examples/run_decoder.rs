use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use huggingface_candle_transformer::transformer::{
    decoder::Decoder,
    masking::{create_seq_mask, get_mask},
    pos_embedding_sin::PosEmbeddingSin,
};
fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let d_model: usize = 64;
    let d_output: usize = 20;
    let ff_d_hidden = 128;
    let num_heads: usize = 8;

    let max_seq_len: usize = 10;
    let pos_embed = PosEmbeddingSin::new(&vb, max_seq_len, d_model)?;

    let model = Decoder::new(&vb, d_model, d_output, num_heads, ff_d_hidden, 3, pos_embed)?;

    let enc_out = Tensor::rand(0.0, 1.0, &[2, 12, d_model], &Device::Cpu)?.to_dtype(DType::F32)?;
    let q = Tensor::rand(0.0, 1.0, &[2, 4, d_output], &Device::Cpu)?.to_dtype(DType::F32)?;
    let src_mask = create_seq_mask(&[2, 2, 2, 2], 12, &Device::Cpu)?;
    let tgt_mask = create_seq_mask(&[2, 2, 2, 2], 4, &Device::Cpu)?;

    let out = model.forward(&q, &enc_out, Some(&tgt_mask))?;

    println!("out shape:{:?}", out.shape());
    println!("out:{:?}", out);

    Ok(())
}
