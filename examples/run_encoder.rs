use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use huggingface_candle_transformer::transformer::{
    encoder::Encoder, masking::create_seq_mask, pos_embedding_sin::PosEmbeddingSin,
};
fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let d_model: usize = 64;
    let d_input: usize = 20;
    let ff_d_hidden = 128;
    let num_heads: usize = 8;
    let num_blocks = 3;

    let max_seq_len: usize = 10;
    let pos_embed = PosEmbeddingSin::new(&vb, max_seq_len, d_model)?;

    let model = Encoder::new(
        &vb,
        d_model,
        d_input,
        num_heads,
        ff_d_hidden,
        num_blocks,
        pos_embed,
    )?;

    let img = Tensor::rand(0.0, 1.0, &[2, 4, d_input], &Device::Cpu)?.to_dtype(DType::F32)?;
    let src_mask = create_seq_mask(&[2, 2, 2, 2], 4, &Device::Cpu)?;
    let tgt_mask = create_seq_mask(&[2, 2, 2, 2], 4, &Device::Cpu)?;

    let out = model.forward(&img, Some(&tgt_mask))?;

    println!("out shape:{:?}", out.shape());
    println!("out:{:?}", out);

    Ok(())
}
