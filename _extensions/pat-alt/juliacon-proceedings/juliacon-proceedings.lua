
-- Define the custom Julia listing environment
function JuliaListing(elem)
  if elem.classes:includes("julia") then
    -- Add your custom Julia listing environment code here
    -- For example, you can use the `lstlisting` environment from the `listings` package
    return pandoc.RawBlock("latex", "\\begin{lstlisting}[language=Julia]\n" .. elem.text .. "\n\\end{lstlisting}")
  end
end

-- Apply the custom Julia listing environment to all code blocks
function CodeBlock(elem)
  if elem.classes:includes("julia") then
    return JuliaListing(elem)
  end
end

-- Register the Lua filter
return {
  { CodeBlock = CodeBlock }
}
