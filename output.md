# Exercise 4

Zishi Zhang
23-741-390

---

## Task 1 LayerNorm in `JoeyNMT`

---

- LayerNorm in Encoder
    - Multi-Head Attention
        Under `./joeynmt/transformer_layers.py` Line `276-283`
        ```python
        if self._layer_norm_position == "pre":
                x = self.layer_norm(x)

            x, _ = self.src_src_att(x, x, x, mask)
            x = self.dropout(x) + self.alpha * residual

            if self._layer_norm_position == "post":
                x = self.layer_norm(x)
        ```

        If the `layer_norm_postion` is `pre`, the layernorm will be execute before self attention layer.
    
    - Positionwise Feed Forward
        Under `./joeynmt/transformer_layers.py` Line `158-164`
        ```python
        if self._layer_norm_position == "pre":
            x = self.layer_norm(x)

        x = self.pwff_layer(x) + self.alpha * residual

        if self._layer_norm_position == "post":
            x = self.layer_norm(x)
        ```

        If the`layer_norm_postion` is `pre`, the layernorm will be execute before feed forward layer.