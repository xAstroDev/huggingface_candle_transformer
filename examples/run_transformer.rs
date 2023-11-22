use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use huggingface_candle_transformer::transformer::{
    masking::{create_seq_mask, get_mask},
    pos_embedding_sin::PosEmbeddingSin,
    transformer::Transformer,
};
fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let d_model: usize = 64;
    let d_input: usize = 15;
    let d_output: usize = 20;
    let ff_d_hidden_enc = 128;
    let ff_d_hidden_dec = 128;
    let num_heads_enc: usize = 8;
    let num_heads_dec: usize = 8;
    let num_blocks_enc = 4;
    let num_blocks_dec = 2;

    let max_seq_len_enc: usize = 15;
    let max_seq_len_dec: usize = 5;

    let pe_enc = PosEmbeddingSin::new(&vb, max_seq_len_enc, d_model)?;
    let pe_dec = PosEmbeddingSin::new(&vb, max_seq_len_dec, d_model)?;

    let model = Transformer::new(
        &vb,
        d_model,
        d_input,
        d_output,
        num_heads_enc,
        num_heads_dec,
        ff_d_hidden_enc,
        ff_d_hidden_dec,
        num_blocks_enc,
        num_blocks_dec,
        pe_enc,
        pe_dec,
    )?;

    let xs = Tensor::rand(0.0, 1.0, &[2, 12, d_input], &Device::Cpu)?.to_dtype(DType::F32)?;

    let q = Tensor::rand(0.0, 1.0, &[2, 4, d_output], &Device::Cpu)?.to_dtype(DType::F32)?;

    let src_mask = create_seq_mask(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 12, &Device::Cpu)?;
    let tgt_mask = create_seq_mask(&[2, 2, 2, 2], 4, &Device::Cpu)?;

    let out = model.forward(&q, &xs, Some(&src_mask), Some(&tgt_mask))?;
    //let out = model.forward(&q, &xs, None, None)?;

    println!("out shape:{:?}", out.shape());
    println!("out:{:?}", out);

    Ok(())
}
