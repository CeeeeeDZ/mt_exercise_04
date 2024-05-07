# Exercise 4

Zishi Zhang
23-741-390

---

## Task 1 LayerNorm in `JoeyNMT`  

---

### Implementations  

- **LayerNorm**  
        Under `./joeynmt/transformer_layers.py` Line `143`
        ```python
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        ```

        The layer normalization is implement by using method from torch called `LayerNorm`, eps is set to 1e-6 which will be add to the denominator for numerical stability. This is common throughout the whole program

- **LayerNorm in Encoder**
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
    
    - LayerNorm of the whole encoder  
        Under `../joeynmt/encoder.py` Line `257-258`  
        ```python
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        ```  
          
        If the`layer_norm_postion` is `pre`, the layernorm will be execute after multi stacks of encoder.

- **LayerNorm in Decoder**
    - Masked Multi-Head Attention
        Under `./joeynmt/transformer_layers.py` Line `377-384`
        ```python
        if self._layer_norm_position == "pre":
            x = self.x_layer_norm(x)

        h1, _ = self.trg_trg_att(x, x, x, mask=trg_mask)
        h1 = self.dropout(h1) + self.alpha * residual

        if self._layer_norm_position == "post":
            h1 = self.x_layer_norm(h1)
        ```

        If the `layer_norm_postion` is `pre`, the layernorm will be execute before masked self attention layer.
    
    - Source-Target Multi-Head Attention
        Under `./joeynmt/transformer_layers.py` Line `388-397`
        ```python
        if self._layer_norm_position == "pre":
            h1 = self.dec_layer_norm(h1)

        h2, att = self.src_trg_att(
            memory, memory, h1, mask=src_mask, return_weights=return_attention
        )
        h2 = self.dropout(h2) + self.alpha * h1_residual

        if self._layer_norm_position == "post":
            h2 = self.dec_layer_norm(h2)
        ```

        If the `layer_norm_postion` is `pre`, the layernorm will be execute before source-target self attention layer.
    
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
    
    - LayerNorm of the whole decoder
        Under `../joeynmt/decoder.py` Line `603-604`
        ```python
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        ```

        If the`layer_norm_postion` is `pre`, the layernorm will be execute after multi stacks of decoder and before the final linear layer.

---

### Default Settings

- **Encoder**
    Under `../joeynmt/encoder.py` Line `208`
    ```python
    layer_norm=kwargs.get("layer_norm", "pre"),
    ```

    Line `216-219`
    ```python
    self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=1e-6)
            if kwargs.get("layer_norm", "post") == "pre" else None
        )
    ```

    If the layer_norm are not specified in config file, the postion will be `pre` by default when calling the `TransformerEncoderLayer` class and `post` when assigning `LayerNorm` method to `self.layer_norm` in the encoder by using `get("layer_norm", "post")`
    
    > This inconsistency might due to negligence by creator of this repository, and it should be like `layer_norm=kwargs.get("layer_norm", "post"),`

- **Decoder**
    Under `../joeynmt/decoder.py` Line `537`
    ```python
    layer_norm=kwargs.get("layer_norm", "post"),
    ```

    Line `543-546`
    ```python
    self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=1e-6)
            if kwargs.get("layer_norm", "post") == "pre" else None
        )
    ```

    If the layer_norm are not specified in config file, the postion will be `post` by default when calling the `TransformerDecoderLayer` class and assigning `LayerNorm` method to `self.layer_norm` in the decoder by using `get("layer_norm", "post")`