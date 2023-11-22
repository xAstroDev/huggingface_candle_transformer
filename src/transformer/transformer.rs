use super::{decoder::Decoder, encoder::Encoder};
use candle_core::Module;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct Transformer<PE: Module> {
    pub encoder: Encoder<PE>,
    pub decoder: Decoder<PE>,
}

impl<PE: Module> Transformer<PE> {
    pub fn new(
        vb: &VarBuilder,
        d_model: usize,
        d_input: usize,
        d_output: usize,
        num_heads_enc: usize,
        num_heads_dec: usize,
        ff_d_hidden_enc: usize,
        ff_d_hidden_dec: usize,
        num_blocks_enc: usize,
        num_blocks_dec: usize,
        pe_enc: PE,
        pe_dec: PE,
    ) -> Result<Self> {
        let encoder = Encoder::new(
            &vb,
            d_model,
            d_input,
            num_heads_enc,
            ff_d_hidden_enc,
            num_blocks_enc,
            pe_enc,
        )?;

        let decoder = Decoder::new(
            &vb,
            d_model,
            d_output,
            num_heads_dec,
            ff_d_hidden_dec,
            num_blocks_dec,
            pe_dec,
        )?;

        let ret = Self {
            encoder: encoder,
            decoder: decoder,
        };

        return Ok(ret);
    }

    pub fn forward(
        &self,
        query: &Tensor,
        xs: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let enc_output = self.encoder.forward(xs, src_mask)?;
        let out = self.decoder.forward(query, &enc_output, tgt_mask);
        return out;
    }
}
