use candle_core::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use super::mh_selfattention::dot_product_attention;

#[derive(Debug)]
pub struct MHAttention {
    pub q: Linear,
    pub kv: Linear,
    pub proj: Linear,
    pub num_heads: usize,
    pub d_model: usize,
    pub scale: f64,
}

impl MHAttention {
    pub fn new(
        vb: &VarBuilder,
        d_model: usize,
        num_heads: usize,
        q_bias: bool,
        kv_bias: bool,
        proj_bias: bool,
    ) -> Result<Self> {
        let q = if q_bias {
            linear(d_model, d_model, vb.pp("q"))
        } else {
            linear_no_bias(d_model, d_model, vb.pp("q"))
        }?;

        let kv = if kv_bias {
            linear(d_model, d_model * 2, vb.pp("kv"))
        } else {
            linear_no_bias(d_model, d_model * 2, vb.pp("kv"))
        }?;

        let proj = if proj_bias {
            linear(d_model, d_model, vb.pp("proj"))
        } else {
            linear_no_bias(d_model, d_model, vb.pp("proj"))
        }?;

        let scale = 1. / ((d_model / num_heads) as f64).sqrt();
        Ok(Self {
            d_model,
            q: q,
            kv: kv,
            proj,
            num_heads,
            scale,
        })
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bq, nq, c) = query.dims3()?;

        let q = (self.q.forward(query)? * self.scale)?;
        let q = q.reshape((bq, self.num_heads, nq, self.d_model / self.num_heads))?;

        let (b, nkv, c) = key_value.dims3()?;

        let kv = self.kv.forward(key_value)?.reshape((
            b,
            nkv,
            2,
            self.num_heads,
            self.d_model / self.num_heads,
        ))?;
        let kv = kv.permute((2, 0, 3, 1, 4))?;
        let k = kv.i(0)?;
        let v = kv.i(1)?;

        let scored_values = dot_product_attention(&q, &k, &v, mask)?;

        let out = self.proj.forward(&scored_values);

        return out;
    }
}
