use glsl_to_spirv;
use std::fs::read_to_string;
use std::fs::remove_file;
use std::fs::File;
use std::io::prelude::*;

fn save_into_spirv(dir: &str, filename: &str) {
    for (ext, ty) in vec![
        (".vert", glsl_to_spirv::ShaderType::Vertex),
        (".frag", glsl_to_spirv::ShaderType::Fragment),
    ]
    .into_iter()
    {
        let path = dir.to_owned() + "/" + filename + ext;
        let out_path = path.clone() + ".spv";
        let code = read_to_string(&path).unwrap();
        let mut file = glsl_to_spirv::compile(&code, ty).unwrap_or_else(|err| {
            eprintln!("compile {}:", path);
            panic!("{}", err)
        });
        let mut buf = vec![];
        file.read_to_end(&mut buf).unwrap();

        remove_file(&out_path).unwrap();
        let mut output = File::create(&out_path).unwrap();
        output.write_all(&mut buf).unwrap();

        println!("successfully compiled: {} -> {}", &path, &out_path);
    }
}

fn main() {
    save_into_spirv("src/data", "triangle");
}
