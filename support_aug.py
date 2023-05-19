import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="The following are examples of MeToo Movement tweets's stance are classified as \"Support\". Support is defined by indicating favor towards the #MeToo movement or its cause. Please write 100 tweets.\n1. An anonymous source claims to have evidence supporting accusations made against an actor for predatory behavior towards young girls\n2. The police are investigating reports from witnesses who allege they saw someone engaging in inappropriate activities with minors\n3. Three separate sources confirm allegations about a well-known leader’s involvement in questionable conduct involving minors\n4. Allegations of misconduct by me are completely untrue - they will be proven as such\n5. My name is being dragged through the mud with these spurious allegations - however, they remain unfounded\n6. I stand with the victims of abuse and support them in their fight for justice \n7. The #MeToo movement is an incredibly important step forward in protecting people from sexual harassment and assault \n8. No one should ever have to experience any form of violence or discrimination because of their gender or identity  \n9. We must work together to end all forms of oppression that keep us from achieving our potential as a society \n10. Let’s put an end to the silence surrounding survivors who are brave enough to speak out against injustice! #MeToo",
  temperature=0.7,
  max_tokens=600,
  top_p=1,
  frequency_penalty=1.45,
  presence_penalty=1.35
)