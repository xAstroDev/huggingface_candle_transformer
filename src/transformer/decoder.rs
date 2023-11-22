use super::decoder_block::DecoderBlock;
use candle_core::Module;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::linear_no_bias;
use candle_nn::Linear;
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct Decoder<PE: Module> {
    pub blocks: Vec<DecoderBlock>,
    pub output_embedding: Linear,
    pub pos_embed: PE,
}

impl<PE: Module> Decoder<PE> {
    pub fn new(
        vb: &VarBuilder,
        d_model: usize,
        d_output: usize,
        num_heads: usize,
        ff_d_hidden: usize,
        num_blocks: usize,
        pe: PE,
    ) -> Result<Self> {
        let mut blocks: Vec<DecoderBlock> = Vec::new();
        for _ in 0..num_blocks {
            let b = DecoderBlock::new(vb, d_model, num_heads, ff_d_hidden)?;
            blocks.push(b);
        }

        let output_embedding = linear_no_bias(d_output, d_model, vb.pp("output_embedding"))?;
        let ret = Decoder {
            blocks: blocks,
            pos_embed: pe,
            output_embedding: output_embedding,
        };

        return Ok(ret);
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        enc_output: &Tensor,
        tgt_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let pxs = self.output_embedding.forward(xs)?;
        let mut out = self.pos_embed.forward(&pxs)?;
        for l in &self.blocks {
            out = l.forward(&out, enc_output, tgt_mask)?;
        }
        return Ok(out);
    }
}
