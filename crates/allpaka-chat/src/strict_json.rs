//! JSON values with duplicate object keys rejected at every nesting level.
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = Unique;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("JSON with unique object keys")
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Unique,E> { Ok(Unique(Value::Bool(value))) }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Unique,E> { Ok(Unique(value.into())) }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Unique,E> { Ok(Unique(value.into())) }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Unique,E> {
                serde_json::Number::from_f64(value).map(|number|Unique(Value::Number(number))).ok_or_else(||E::custom("Nonfinite JSON number"))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Unique,E> { Ok(Unique(Value::String(value.into()))) }
            fn visit_string<E: de::Error>(self, value: String) -> Result<Unique,E> { Ok(Unique(Value::String(value))) }
            fn visit_unit<E: de::Error>(self) -> Result<Unique,E> { Ok(Unique(Value::Null)) }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Unique,A::Error> {
                let mut values=Vec::new();
                while let Some(Unique(value))=seq.next_element()? { values.push(value); }
                Ok(Unique(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Unique,A::Error> {
                let mut values=serde_json::Map::new();
                while let Some(key)=map.next_key::<String>()? {
                    if values.contains_key(&key) { return Err(de::Error::custom("Duplicate JSON key")); }
                    let Unique(value)=map.next_value()?;
                    values.insert(key,value);
                }
                Ok(Unique(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}
pub(crate) fn parse(text: &str) -> Result<Value,serde_json::Error> {
    serde_json::from_str::<Unique>(text).map(|value|value.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nested_and_escaped_duplicate_keys() {
        for text in [r#"{"a":1,"a":2}"#,r#"[{"nested":{"x":true,"x":false}}]"#,r#"{"a":1,"\u0061":2}"#] { assert!(parse(text).is_err()); }
        for text in [r#"{"a":{"x":1},"b":{"x":2}}"#,r#"[true,null,2,-3,1.5,"text"]"#] {
            assert_eq!(parse(text).unwrap(),serde_json::from_str::<Value>(text).unwrap());
        }
    }
}
