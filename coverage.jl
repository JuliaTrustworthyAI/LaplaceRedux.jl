using Pkg
using Coverage

Pkg.test("LaplaceRedux", coverage=true)

coverage = process_folder()
coverage = append!(coverage, process_folder("test"))

open("coverage/lcov.info", "w") do io
    LCOV.write(io, coverage)
end;

run(`genhtml coverage/lcov.info --branch-coverage -o coverage/output/`)
