use super::{
    add_norm::AddNorm, mh_attention::MHAttention, mh_selfattention::MHSelfAttention,
    positionwise_ffn::PositionWiseFFN,
};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct DecoderBlock {
    pub self_attn: MHSelfAttention,
    pub cross_attn: MHAttention,
    pub feed_forward: PositionWiseFFN,
    pub addnorm1: AddNorm,
    pub addnorm2: AddNorm,
    pub addnorm3: AddNorm,
}

impl DecoderBlock {
    pub fn new(
        vb: &VarBuilder,
        d_model: usize,
        num_heads: usize,
        ff_d_hidden: usize,
    ) -> Result<Self> {
        let sa = MHSelfAttention::new(vb, d_model, num_heads, false, false)?;
        let ca = MHAttention::new(vb, d_model, num_heads, false, false, false)?;
        let ff = PositionWiseFFN::new(vb, d_model, ff_d_hidden)?;
        let n1 = AddNorm::new(vb, d_model)?;
        let n2 = AddNorm::new(vb, d_model)?;
        let n3 = AddNorm::new(vb, d_model)?;

        let ret = DecoderBlock {
            addnorm1: n1,
            addnorm2: n2,
            addnorm3: n3,
            self_attn: sa,
            feed_forward: ff,
            cross_attn: ca,
        };

        return Ok(ret);
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        enc_output: &Tensor,
        tgt_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attn_out = self.self_attn.forward(xs, tgt_mask)?;
        let out = self.addnorm1.forward(xs, &attn_out)?;
        let cross_attn_out = self.cross_attn.forward(&out, enc_output, None)?;
        let out = self.addnorm2.forward(&out, &cross_attn_out)?;

        let ff_out = self.feed_forward.forward(&out)?;
        let out = self.addnorm3.forward(&out, &ff_out);
        return out;
    }
}
