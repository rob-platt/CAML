# About my site

<!--These are called Admonitions-->
!!! warning
    Don't use this site, its trash.
    
!!! quote
    "Reproduceability is optional, not a requirement" - Myself

<!--Can use ? instead of ! for dropdown-->
??? danger
    Are you sure you want to jump down this rabbit hole?

<!--Can add codeblocks, add python for language specific highlighting-->
```python
import pandas as pd # (1)!
df = pd.read_json("MyFilepath")
```
<!--Can use (1) and add references - called annotations-->
1.  Convention

<!--Standard markdown heading options-->
### Bugs
<!--Can use === to get multiple tabs-->
=== "Bug in Python"
    This one is a big un.
    ```python
    import n2n4m # this is convention not required
    raise TypeError("Why would you do that?")
    ```

=== "Bug 2"
    This is annoying but really you're a dumbass if you cause this.

<div class="grid cards" markdown>
- :fontawesome-brands-python: Python
- :fontawesome-brands-github: Github
</div>
