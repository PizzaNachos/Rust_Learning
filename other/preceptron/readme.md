https://rustwasm.github.io/docs/wasm-bindgen/
https://rustwasm.github.io/docs/wasm-bindgen/examples/webgl.html

Structure:
    Rust based Nueral Network struct
        JS calls into rust with an array of numbers to test
        Rust runs all calculations and returns a pointer to the start of the memory where stuff is returned

        We do the experimental training in WASM too, try and make this threaded somehow
        make out polynomials in rust -> when they're called rust lazy loads and then caches the entire
        polynomial returned with the numbers passed in, JS renders in as an animation frame

        Two versions:
            In CPU only
                Genetic vs Back propigation
            webGL backend for the NN stuff
                Genetic vs Back propigation

