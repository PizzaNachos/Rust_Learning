import init, {
  Network,
  random_f64_0_1,
  FunctionTuple,
} from "../pkg/preceptron.js";
import { calc_function_space } from "./function_space.js";

// Use ES module import syntax to import functionality from the module
// that we have compiled.
//
// Note that the `default` import is an initialization function which
// will "boot" the module and make it ready to use. Currently browsers
// don't support natively imported WebAssembly as an ES module, but
// eventually the manual initialization won't be required!
function draw_nn(n1, c_w, c_h, function_space, low_res_scale) {
    const canvas2 = document.getElementById("canvas2");
    const ctx2 = canvas2.getContext("2d");

    ctx2.fillStyle = "rgb(0, 0, 0)";
    ctx2.fillRect(0, 0, c_w, c_h);
    for (let i = 0; i < c_w / low_res_scale; i++) {
        for (let j = 0; j < c_h / low_res_scale; j++) {
        let g = n1.feed_forward([function_space[i][j].x, function_space[i][j].y]);
        ctx2.fillStyle = `rgb(${g[0] * 255}, ${g[1] * 255}, ${g[2] * 255})`;
        ctx2.fillRect(
            i * low_res_scale,
            j * low_res_scale,
            low_res_scale,
            low_res_scale
        );
        }
    }
}
function draw_poly(polys, c_w, c_h, function_space, low_res_scale) {
    const canvas2 = document.getElementById("canvas1");
    const ctx2 = canvas2.getContext("2d");

    ctx2.fillStyle = "rgb(0, 0, 0)";
    ctx2.fillRect(0, 0, c_w, c_h);
    for (let i = 0; i < c_w / low_res_scale; i++) {
        for (let j = 0; j < c_h / low_res_scale; j++) {
        let x = function_space[i][j].x;
        let y = function_space[i][j].y;

        let r = polys.red(x, y);
        let g = polys.green(x, y);
        let b = polys.blue(x, y);

        ctx2.fillStyle = `rgb(${r * 255}, ${g * 255}, ${b * 255})`;
        ctx2.fillRect(
            i * low_res_scale,
            j * low_res_scale,
            low_res_scale,
            low_res_scale
        );
        }
    }
}
async function run() {
    // First up we need to actually load the wasm file, so we use the
    // default export to inform it where the wasm file is located on the
    // server, and then we wait on the returned promise to wait for the
    // wasm to be loaded.
    //
    // It may look like this: `await init('./pkg/without_a_bundler_bg.wasm');`,
    // but there is also a handy default inside `init` function, which uses
    // `import.meta` to locate the wasm file relatively to js file.
    //
    // Note that instead of a string you can also pass in any of the
    // following things:
    //
    // * `WebAssembly.Module`
    //
    // * `ArrayBuffer`
    //
    // * `Response`
    //
    // * `Promise` which returns any of the above, e.g. `fetch("./path/to/wasm")`
    //
    // This gives you complete control over how the module is loaded
    // and compiled.
    //
    // Also note that the promise, when resolved, yields the wasm module's
    // exports which is the same as importing the `*_bg` module in other
    // modes
    await init();
    let c_w = 500;
    let c_h = 500;
    let low_res_scale = 4;
    let function_space = calc_function_space(c_w, c_h, 20, 20, low_res_scale);

    let function_tuple = FunctionTuple.new();
    draw_poly(function_tuple, c_w, c_h, function_space, low_res_scale);

    let n = Network.new([2,3,3]);


    for (let i = 0; i < 1; i++) {
        let xs = [];
        let ys = [];

        let rs =  [];
        let gs = [];
        let bs =  [];

        for (let i = 0; i < c_w / low_res_scale; i++) {
            for (let j = 0; j < c_h / low_res_scale; j++) {
                let x = function_space[i][j].x;
                let y = function_space[i][j].y;

                xs.push(x)
                ys.push(y)

                rs.push( function_tuple.red(x, y))
                gs.push( function_tuple.green(x, y))
                bs.push( function_tuple.blue(x, y))

                // let poly_output = [r,g,b];
                // let neuron_output = n.feed_forward([x,y]);


                // console.log(error);
            }
        }

        // console.log(xs,ys,rs,gs,bs);

        // console.log();
        // let x = n.back_propigate_js(xs, ys, rs,gs,bs, 0.0005)
        // console.log(x);

        // let x = n.back_propigate_js([0,0], [0,0], [1,1],[1,1],[1,1], 0.0005)
        let x = n.back_propigate_js([0], [0], [.1],[.25],[.75], 0.0005)

        console.log(x);

        draw_nn(n, c_w, c_h, function_space, low_res_scale);
        await new Promise((resolve) => setTimeout(resolve, 100));

        // let red_error_number = 0;
        // let green_error_number = 0;
        // let blue_error_number = 0;
        // let e = agregate_error;
        // for(let i = 0; i < e[0].length; i++){
        //     red_error_number += (e[0][i] / e[0].length)
        //     green_error_number += (e[1][i] / e[1].length)
        //     blue_error_number += (e[2][i] / e[2].length)
        // }
        // // console.log(red_error_number, green_error_number, blue_error_number);
        // try{
        // } catch(e){
        //     console.log(e);
        //     break;
        // }
        // console.log(n.get_last_bias());
        // // n.tw(0.01);
        // let tweak_ammount = 0.1;
        // n.tweak_random_weight(tweak_ammount);
        // tweak_ammount += .001;
    }
    // And afterwards we can use all the functionality defined in wasm.
    // const result = add(1, 2);
    // console.log(`1 + 2 = ${result}`);
    // if (result !== 3)
    //   throw new Error("wasm addition doesn't work!");
}

run();
