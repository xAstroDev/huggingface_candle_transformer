use crate::transformer::masking::masked_softmax;
use candle_core::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

pub fn dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let attn = q.matmul(&k.t()?)?;
    let sm_masked_attn = masked_softmax(attn, mask)?;
    //println!("attn:{:?}", sm_masked_attn.shape());

    let scored_values = sm_masked_attn.matmul(&v)?;
    //println!("scored_values:{:?}", scored_values.shape());

    let scored_values = scored_values.permute((0, 2, 1, 3))?;
    //println!("scored_values:{:?}", scored_values.shape());

    let dims = scored_values.dims();

    let scored_values = scored_values.reshape((dims[0], dims[1], dims[2] * dims[3]));
    //println!("scored_values:{:?}", scored_values?.shape());
    return scored_values;
}

#[derive(Debug)]
pub struct MHSelfAttention {
    pub qkv: Linear,
    pub proj: Linear,
    pub num_heads: usize,
    pub d_model: usize,
    pub scale: f64,
}

impl MHSelfAttention {
    pub fn new(
        vb: &VarBuilder,
        d_model: usize,
        num_heads: usize,
        qkv_bias: bool,
        proj_bias: bool,
    ) -> Result<Self> {
        let qkv = if qkv_bias {
            linear(d_model, d_model * 3, vb.pp("qkv"))
        } else {
            linear_no_bias(d_model, d_model * 3, vb.pp("qkv"))
        }?;

        let proj = if proj_bias {
            linear(d_model, d_model, vb.pp("proj"))
        } else {
            linear_no_bias(d_model, d_model, vb.pp("proj"))
        }?;

        let scale = 1. / ((d_model / num_heads) as f64).sqrt();
        Ok(Self {
            d_model: d_model,
            qkv: qkv,
            proj: proj,
            num_heads: num_heads,
            scale: scale,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;

        let qkv = self.qkv.forward(xs)?;

        let qkv = qkv.reshape((b, n, 3, self.num_heads, self.d_model / self.num_heads))?;

        let qkv = qkv.permute((2, 0, 3, 1, 4))?;

        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let scored_values = dot_product_attention(&q, &k, &v, mask)?;

        let out = self.proj.forward(&scored_values);
        return out;
    }
}
