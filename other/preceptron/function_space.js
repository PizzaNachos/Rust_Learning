export function calc_function_space(c_w = 500, c_h = 500, p_w = 20, p_h = 20, low_res_scale = 4){
    let function_space = [];

    for(let i = 0; i < c_w / low_res_scale; i++){
      function_space.push([]);
        for(let j = 0; j < c_h / low_res_scale; j++){
            let x = (i / (c_w / (p_w * low_res_scale))) - (p_w / 2);
            let y = (j / (c_h / (p_h * low_res_scale))) - (p_h / 2);
            function_space[i].push({x:x, y:y});
            // let g = n1.feed_forward([x,y]);
            // ctx2.fillStyle = `rgb(${g[0] * 255}, ${g[1] * 255}, ${g[2] * 255})`;
            // ctx2.fillRect(i * low_res_scale, j * low_res_scale, low_res_scale, low_res_scale);
        }
    }
    return function_space
}